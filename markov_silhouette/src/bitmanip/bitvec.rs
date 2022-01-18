use smallvec::{SmallVec, smallvec};
use serde::{Serialize, Deserialize};

type PackedType = u64;

const PACKED_TYPE_BYTES: usize = std::mem::size_of::<PackedType>();
const PACKED_TYPE_BITS: usize = PACKED_TYPE_BYTES * 8;

const fn num_blocks(len: usize) -> usize {
    len / PACKED_TYPE_BITS + (len % PACKED_TYPE_BITS > 0) as usize
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Hash, Default, Serialize, Deserialize)]
pub struct BitVec {
    len: usize,
    vec: SmallVec<[PackedType; 2]>,
}

impl BitVec {
    /// Create BitArray with all ones
    pub fn new_ones(len: usize) -> Self {
        Self {
            len,
            vec: smallvec![PackedType::MAX; num_blocks(len)],
        }
    }

    /// Create BitArray with all zeros
    pub fn new_zeros(len: usize) -> Self {
        Self {
            len,
            vec: smallvec![0; num_blocks(len)],
        }
    }

    pub fn new_val(len: usize, val: bool) -> Self {
        if val {
            Self::new_ones(len)
        } else {
            Self::new_zeros(len)
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Get specific bit, with no safety checks
    pub unsafe fn get_unchecked(&self, i: usize) -> bool {
        let b = i / PACKED_TYPE_BITS;
        let n = i % PACKED_TYPE_BITS;
        (self.vec.get_unchecked(b) >> n) & 0b1 == 1
    }

    /// Get specific bit
    pub fn get(&self, i: usize) -> Option<bool> {
        if i >= self.len() {
            None
        } else {
            Some(unsafe { self.get_unchecked(i) })
        }
    }

    /// Set specific bit, with no safety checks
    /// Returns old value
    pub unsafe fn set_unchecked(&mut self, i: usize, val: bool) -> bool {
        let b = i / PACKED_TYPE_BITS;
        let n = i % PACKED_TYPE_BITS;

        let old = self.get_unchecked(i);
        self.vec[b] = (self.vec[b] & !(1 << n)) | ((val as u64) << n);
        return old;
    }

    /// Set specific bit.
    /// Returns old value
    pub fn set(&mut self, i: usize, val: bool) -> Option<bool> {
        if i >= self.len() {
            None
        } else {
            Some(unsafe { self.set_unchecked(i, val) })
        }
    }
}

impl std::fmt::Binary for BitVec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.len() {
            unsafe {
                write!(
                    f,
                    "{}{}",
                    if i > 0 && i % 8 == 0 { " " } else { "" },
                    if self.get_unchecked(i) { 1 } else { 0 }
                )?
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_blocks0() {
        assert_eq!(num_blocks(0), 0);
    }

    #[test]
    fn test_num_blocks1() {
        assert_eq!(num_blocks(1), 1);
    }

    #[test]
    fn test_num_blocks30() {
        assert_eq!(num_blocks(30), 1);
    }

    #[test]
    fn test_num_blocks64() {
        assert_eq!(num_blocks(64), 1);
    }

    #[test]
    fn test_num_blocks65() {
        assert_eq!(num_blocks(65), 2);
    }
}
