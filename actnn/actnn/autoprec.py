import torch
import numpy as np
import random
import actnn.cpp_extension.calc_precision as ext_calc_precision

# Automatically compute the precision for each tensor


class AutoPrecision:
    def init_from_dims(self, dims):
        self.dims = torch.tensor(dims, dtype=torch.long)
        self.L = self.dims.shape[0]
        # Sensitivity for each tensor, tied within each group
        self.C = torch.ones(self.L)
        self.bits = torch.ones(self.L, dtype=torch.int32) * self.abits
        self.total_bits = self.abits * self.dims.sum()
        self.order = torch.randperm(self.L)

    def __init__(self, model, quantizer, bits, max_bits=8, momentum=0.99,
                 adapt_interval=100, warmup_iter=100):
        self.model = model
        self.quantizer = quantizer

        self.dims = None

        self.abits = bits
        self.max_bits = max_bits
        self.perm = []

        self.initialized = False

        # For maintaining batch_grad and detecting overly large quantization variance
        self.momentum = momentum
        self.adapt_interval = 0
        self.warmpup_iter = warmup_iter
        self.beta1 = 1e-7
        self.batch_grad = 0
        self.grad_var = 0
        self.adapt_interval = adapt_interval

        self.iter = 0

        # self.refresh_bits()

    def before_iter(self):
        # self.seeds = [random.getstate(), np.random.get_state(), torch.get_rng_state()]
        random.seed(self.iter)
        np.random.seed(self.iter)
        torch.manual_seed(self.iter)
        for l in range(self.L):
            self.quantizer.inject_noises[l] = False

    def iterate_wrapper(self, backprop):
        def backprop_iter():
            ret = backprop()
            self.quantizer.iterate()
            return ret

        if self.dims is None:
            self.init_from_dims(self.quantizer.dims)
        self.before_iter()
        loss, output, grad = backprop_iter()
        self.iterate(grad, backprop_iter)

    def iterate(self, det_grad, backprop):
        # TODO det_grad is actually not necessary
        def get_grad():
            # TODO this is somewhat tricky...
            # The noise should be injected with other random seeds
            # TODO setstate & getstate won't work, why?
            # random.setstate(self.seeds[0])
            # np.random.set_state(self.seeds[1])
            # torch.set_rng_state(self.seeds[2])
            # torch.use_deterministic_algorithms(True)
            random.seed(self.iter)
            np.random.seed(self.iter)
            torch.manual_seed(self.iter)

            backprop()

            # random.setstate(self.seeds[0])
            # np.random.set_state(self.seeds[1])
            # torch.set_rng_state(self.seeds[2])

            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.append(param.grad.ravel())

            return torch.cat(grad, 0)

        if self.iter == 0:
            # Do full adaptation
            print('ActNN: Initializing AutoPrec...')
            # sum_c = 0
            for l in range(self.L):
                self.quantizer.inject_noises[l] = True
                grad = get_grad()
                self.C[l] = ((det_grad - grad) ** 2).sum() * 4
                self.quantizer.inject_noises[l] = False

                # sum_c = sum_c + self.C[l]
            # print('C sum ', sum_c)
        elif self.iter % self.adapt_interval == 0:
            if len(self.perm) == 0:
                self.perm = torch.randperm(self.L)
            l = self.perm[-1]
            self.perm = self.perm[:-1]

            self.quantizer.inject_noises[l] = True
            grad = get_grad()
            self.C[l] = ((det_grad - grad) ** 2).sum() * \
                4  # Hack: always use 2bit
            self.quantizer.inject_noises[l] = False

        self.iter += 1

        # Maintain batch grad
        momentum = self.momentum
        self.beta1 = self.beta1 * momentum + 1 - momentum
        self.batch_grad = self.batch_grad * \
            momentum + (1 - momentum) * det_grad
        bgrad = self.batch_grad / self.beta1
        gvar = ((bgrad - det_grad)**2).sum()
        self.grad_var = self.grad_var * momentum + (1 - momentum) * gvar

        self.refresh_bits()

    def refresh_bits(self):
        total_bits = self.total_bits

        self.bits = torch.ones(self.L, dtype=torch.int32) * self.max_bits
        self.bits = ext_calc_precision.calc_precision(self.bits,
                                                      self.C,
                                                      self.dims,
                                                      total_bits)

        self.quantizer.bits = [bit.item() for bit in self.bits]
        print("Auto precision bits", self.bits)
        # Warning if the quantization variance is too large
        if self.iter > self.warmpup_iter:
            overall_var = self.grad_var / self.beta1
            quantization_var = (
                self.C * 2 ** (-2 * self.bits.float())).sum().cuda()
            if quantization_var > overall_var * 0.1:
                print('ActNN Warning: Quantization variance is too large. Consider increasing number of bits.',
                      quantization_var, overall_var)