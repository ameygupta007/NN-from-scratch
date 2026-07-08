class Value:

    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self._prev = set(children)
        self._op = op
        self.grad = 0
        self.label = label

    def __repr__(self):
        return f"(Value:{self.data})"
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        return Value(self.data + other.data, (self, other), '+')
     
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        return Value(self.data * other.data, (self, other), '*')

    def __rmul__(self, other):
        return self * other
    

def main():
    pass

if __name__ == "__main__":
    main()