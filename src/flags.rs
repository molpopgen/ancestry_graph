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
    struct PropagationBitFlags: u32 {
        const EMPTY = 0;
        const KEEP_UNARY_NODES = 1 << 1;
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
pub struct PropagationOptions(PropagationBitFlags);

impl PropagationOptions {
    pub fn with_keep_unary_nodes(self) -> Self {
        Self(self.0 | PropagationBitFlags::KEEP_UNARY_NODES)
    }

    pub fn keep_unary_nodes(&self) -> bool {
        self.0.contains(PropagationBitFlags::KEEP_UNARY_NODES)
    }
}

#[cfg(test)]
mod test_simplification_flags {
    use super::PropagationOptions;
    #[test]
    fn test_keep_unary() {
        let flags = PropagationOptions::default().with_keep_unary_nodes();
        assert!(flags.keep_unary_nodes())
    }
}
