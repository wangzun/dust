pub trait FastStack<T: Copy + Default> {
    fn push(&mut self, v: T);
    fn pop_fast(&mut self) -> T;
    fn pop(&mut self) -> Option<T>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn clear(&mut self);
}

/// A stack data structure implemented on the stack with fixed capacity.
pub struct StackStack<T: Copy + Default, const STACK_SIZE: usize> {
    data: [T; STACK_SIZE],
    index: usize,
}

impl<T: Copy + Default, const STACK_SIZE: usize> Default for StackStack<T, STACK_SIZE> {
    fn default() -> Self {
        Self {
            data: [Default::default(); STACK_SIZE],
            index: Default::default(),
        }
    }
}

impl<T: Copy + Default, const STACK_SIZE: usize> FastStack<T> for StackStack<T, STACK_SIZE> {
    /// Pushes a value onto the stack. If the stack is full it will overwrite the value in the last position.
    #[inline(always)]
    fn push(&mut self, v: T) {
        let index = if self.index < STACK_SIZE {
            self.index
        } else {
            STACK_SIZE - 1
        };
        self.data[index] = v;
        if self.index + 1 < STACK_SIZE {
            self.index += 1;
        } else {
            self.index = STACK_SIZE - 1;
        }
    }
    /// Pops a value from the stack without checking bounds. If the stack is empty it will return the value in the first position.
    #[inline(always)]
    fn pop_fast(&mut self) -> T {
        if self.index > 0 {
            self.index -= 1;
        }
        self.data[self.index]
    }
    /// Pops a value from the stack.
    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        if self.index > 0 {
            self.index -= 1;
            Some(self.data[self.index])
        } else {
            None
        }
    }
    /// Returns the number of elements in the stack.
    #[inline(always)]
    fn len(&self) -> usize {
        self.index
    }
    /// Returns true if the stack is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.index == 0
    }
    /// Clears the stack, removing all elements.
    #[inline(always)]
    fn clear(&mut self) {
        self.index = 0;
    }
}
