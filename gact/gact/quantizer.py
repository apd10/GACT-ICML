import torch
from torch.overrides import TorchFunctionMode
from gact.conf import config
from gact.ops import op_quantize, op_dequantize, op_quantize_mask, op_dequantize_mask
from gact.utils import uniform_sample, compute_tensor_bytes

P = 9000011
current_func = None
class set_current_func(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        global current_func, current_args

        if not kwargs:
            kwargs = {}
        current_func = func
        out = func(*args, **kwargs)
        current_func = None
        return out

class Quantizer:
    """
    default_bit: the number of bits used to quantize
    swap: if turned on, swap activation memory to CPU
    prefetch: if turned on, activation of the previous layer will be prefetched. the parameter is meaningful only when swap is True
    """

    def __init__(self, default_bit, swap, prefetch):
        self.unrelated_tensors = set()  # record the tensors that should not be quantized
        self.default_bit = default_bit

        self.swap = swap
        if swap:
            self.swap_out_stream = torch.cuda.Stream()
            self.swap_in_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.prefetch = prefetch
        if prefetch:
            self.start_prefetch_event = torch.cuda.Event(blocking=True)
            self.end_prefetch_event = torch.cuda.Event(blocking=True)
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True

        # data collected for auto precision
        self.seeds = {}
        self.bits = {}
        self.dims = {}
        self.dim_shape = {}
        self.function_types = {}
        self.function_based_memory = {}
        self.use_rp = {}
        

        self.iter = 0  # total number of iterations, including the extra inter for auto precision
        # iteration for seed, share the same seed_iter for the same auto precision adaptive step
        self.seed_iter = 0

        #linear stats
        self.linear_packs = 0

    def filter_tensors(self, pairs):
        for _, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    # return should_be_quantized, is_dropout_mask
    # treat dropout mask differently because it can be quantized with 1 bit with a specialized kernel
    def check_quantize(self, input_tensor):
        # does not quantize parameters
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False, False
        # special check for saved mask
        if input_tensor.numel() > 0 and input_tensor.dtype == torch.uint8:
            if (input_tensor.max() == 1) and (input_tensor.min() == 0):
                return True, True
            return False, False
        # only quantize float16 and float32
        if input_tensor.dtype not in [torch.float32, torch.float16]:
            return False, False
        # only quantize activation that requires gradient
        # for example: BN statistics (running mean/var) should not be quantized
        if input_tensor.requires_grad is False:
            return False, False
        # only quantize 2/3/4D tensors for now
        if ((len(input_tensor.shape) != 2)
            and (len(input_tensor.shape) != 3)
            and (len(input_tensor.shape) != 4)
            ):
            return False, False
        return True, False

    def __del__(self):
        print("Linear packs", self.linear_packs)
        del self.ptr_qtensor_map
        del self.layer_key_map
        del self.unrelated_tensors

    def iterate(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True
        self.iter += 1

    def generate_tensor_key(self, t, tid):
        if config.check_dup:
            # sample 100 elements data pointer + tensor.sum() as the key
            sample_cnt = min(100, t.numel())
            key = uniform_sample(t, sample_cnt, add_dataptr=True)
            key.append(t.sum().item())
            return tuple(key)
        else:
            return (tid)

    def quantize(self, input):
        quantize, is_dropout_mask = self.check_quantize(input)


        if not quantize:
            return False, input

        if self.iter == 0:
            if current_func is not None:
                  name = current_func.__name__
            else:
                  name = 'none'
            
            if  name in self.function_based_memory.keys():
                self.function_based_memory[name]  += input.numel()
            else:
                self.function_based_memory[name]  = input.numel()

        # special case: use 1 bit to quantize dropout mask
        if is_dropout_mask:
            q_inputs = op_quantize_mask(input)
            return True, is_dropout_mask, q_inputs

        tid = self.tid
        self.tid += 1
        input_shape = input.shape

        key = self.generate_tensor_key(input, tid)
        self.layer_key_map[tid] = key
        skip_quantize = key in self.ptr_qtensor_map
        islinear = False

        if not skip_quantize:
            if self.iter == 0:
                bit = self.default_bit
                self.bits[tid] = bit
                self.dims[tid] = input.numel()
                self.dim_shape[tid] = input.shape
                self.seeds[tid] = tid
                self.use_rp[tid] = False
                if current_func is not None:
                    self.function_types[tid] = current_func.__name__
                else:
                    self.function_types[tid] = 'none'
            else:
                bit = self.bits[tid]
            # quantize
            #if self.bits[tid] == 1 and self.function_types[tid] in ['linear']:
            if self.use_rp[tid]:
                #print("linear pack")
                q_inputs = self.pack_linear(input, float(bit)/32, self.seeds[tid] + self.seed_iter)
                islinear = True
            else:
                #if self.function_types[tid] == 'relu':
                #    input = (input > 0).type(torch.float)
                assert(bit >= 1)
                q_inputs = op_quantize(
                    input, int(bit), self.seeds[tid] + self.seed_iter)
            if self.swap:
                #  with torch.cuda.stream(self.swap_out_stream):
                # self.swap_out_stream.wait_stream(self.compute_stream)
                q_input_cpu = torch.empty(
                    q_inputs[0].shape,
                    dtype=q_inputs[0].dtype,
                    device="cpu",
                    pin_memory=True,
                )
                q_input_cpu.copy_(q_inputs[0], non_blocking=True)
                q_input_gpu = q_inputs[0]
                del q_input_gpu
                q_inputs[0] = q_input_cpu
            self.ptr_qtensor_map[key] = [q_inputs, 1, tid]
        else:
            # increase the ref count
            self.ptr_qtensor_map[key][1] += 1
        return True, is_dropout_mask, key, input_shape, tid, islinear

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]

        is_dropout_mask = input[1]
        if is_dropout_mask:
            _, is_dropout_mask, q_inputs = input
            ret = op_dequantize_mask(q_inputs)
            return ret

        _, _, key, input_shape, tid, islinear = input
        q_inputs, ref_cnt, key_tid = self.ptr_qtensor_map[key]

        if self.start_bwd and self.swap:
            self.compute_stream.wait_stream(self.swap_out_stream)
            self.start_bwd = False

        # compute waits until prefetch finishes
        if self.prefetch and self.swap:
            self.end_prefetch_event.wait(self.compute_stream)

        if not q_inputs[0].is_cuda:
            q_inputs[0] = q_inputs[0].cuda(non_blocking=False)

        # prefetch previous layer
        if self.prefetch and self.swap:
            # event: start_prefetch
            self.start_prefetch_event.record()
            with torch.cuda.stream(self.swap_in_stream):
                if tid > 0:
                    self.start_prefetch_event.wait(self.swap_in_stream)
                    previous_key = self.layer_key_map[tid - 1]
                    if previous_key in self.ptr_qtensor_map:
                        q_previous_inputs, _, _ = self.ptr_qtensor_map[previous_key]
                        if not q_previous_inputs[0].is_cuda:
                            q_previous_inputs[0] = q_previous_inputs[0].cuda(
                                non_blocking=True
                            )
                    self.end_prefetch_event.record()


        if islinear:
            #print("linear unpack" )
            ret = self.unpack_linear(q_inputs)
        else:
            ret = op_dequantize(q_inputs, input_shape)

        ref_cnt -= 1
        if ref_cnt < 0:
            print("[Error] Ref count < 0", key, ref_cnt)
            exit(-1)
        elif ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret



    def pack_linear(self, x, compression_factor, seed):
        ''' this function is to pack saved variables for linear layer 
            assumes that model weights have already been filtered before this 
        '''
        orig_dtype = x.dtype
        #x = x.half()
        self.linear_packs += 1
        gen = torch.Generator()
        gen.manual_seed(seed)
        random = torch.randint(0, P, (4,), generator=gen)
        A,B,C,D = random[0], random[1], random[2], random[3]
        #print("random", seed, A, B, C, D)
        shape = x.shape # this is a 2d thing
        assert(len(shape) == 2)
        cshape = (int(shape[0] * compression_factor), shape[1])
        #print("Pack Linear", shape, cshape, compression_factor)
        
        compressed_x = torch.zeros(cshape, dtype=x.dtype, device=x.device)

        for i in range(int((shape[0] + cshape[0] - 1)/ cshape[0])):
            idx = ((A*i + B) % P % cshape[0])
            g = 2*((C*i + D) % P % 2) - 1
            #print(i, idx, g)
            
            start = i*cshape[0]
            tlen = min((i+1)*cshape[0], shape[0]) - start
            p1_len = min(cshape[0] - idx, tlen)
            p2_len = tlen - p1_len

            #p1_len at idx
            compressed_x[idx:idx+p1_len] += g * x[start:start+p1_len]
            compressed_x[0:p2_len] += g * x[start+p1_len:start+tlen]

            #print(i, g, idx, idx+p1_len, "<--  x[]", start, start+p1_len)
            #print(i, g, 0, p2_len, "<--  x[]", start+p1_len, start+tlen)
            #print(compressed_x.view(-1))

        #print("PACK LINEAR", id(current_func), x.shape, "-->", compressed_x.shape)
        del x

        return (compressed_x, shape, A, B, C, D, orig_dtype)

    def unpack_linear(self, x):
        ''' this function is to pack saved variables for linear layer 
            assumes that model weights have already been filtered before this 
        '''
        compressed_x, shape, A, B, C, D, orig_dtype = x
        #print("UNPACK LINEAR", id(current_func), compressed_x.shape)
        cshape = compressed_x.shape
        x = torch.zeros(shape, dtype=compressed_x.dtype, device=compressed_x.device)
        for i in range(int((shape[0] + cshape[0] - 1)/ cshape[0])):
            idx = ((A*i + B) % P % cshape[0])
            g = 2*((C*i + D) % P % 2) - 1
            
            start = i*cshape[0]
            tlen = min((i+1)*cshape[0], shape[0]) - start
            p1_len = min(cshape[0] - idx, tlen)
            p2_len = tlen - p1_len

            #p1_len at idx
            x[start:start+p1_len] = compressed_x[idx:idx+p1_len] * g 
            x[start+p1_len:start+tlen] = compressed_x[0:p2_len]* g
            #print(i, g, idx, idx+p1_len, "-->  x[]", start, start+p1_len)
            #print(i, g, 0, p2_len, "-->  x[]", start+p1_len, start+tlen)
            #print(x.view(-1))
        
        del compressed_x
        return x.type(orig_dtype)
