// This module encapsulates the bitflags API
// so that we don't leak details that may
// affect semver later on.

use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Default, Debug, Copy, Clone)]
    struct NodeBitFlags: u32 {
        const EMPTY = 0;
        const IS_SAMPLE = 1 << 1;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Default, Debug, Copy, Clone)]
    struct SimplificationBitOptions: u32 {
        const EMPTY = 0;
        const KEEP_UNARY = 1 << 1;
    }
}

#[repr(transparent)]
#[derive(Default, Debug, Copy, Clone)]
pub struct NodeFlags(NodeBitFlags);

impl NodeFlags {
    pub fn sample() -> Self {
        Self(NodeBitFlags::IS_SAMPLE)
    }
}

#[repr(transparent)]
#[derive(Default, Debug, Copy, Clone)]
pub struct SimplificationOptions(SimplificationBitOptions);

impl SimplificationOptions {
    pub fn with_keep_unary(self) -> Self {
        Self(self.0 | SimplificationBitOptions::KEEP_UNARY)
    }

    pub fn keep_unary(&self) -> bool {
        self.0.contains(SimplificationBitOptions::KEEP_UNARY)
    }
}

#[cfg(test)]
mod test_simplification_flags {
    use super::SimplificationOptions;
    #[test]
    fn test_keep_unary() {
        let flags = SimplificationOptions::default().with_keep_unary();
        assert!(flags.keep_unary())
    }
}
