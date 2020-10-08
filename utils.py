class Utils:
    @staticmethod
    def get_derivative(activation, x):
        if activation == 'sigmoid':
            return x * (1-x)
        elif activation == 'relu':
            if x >= 0:
                return 1
            else:
                return 0
        else:
            raise Exception("Undefined activation function")