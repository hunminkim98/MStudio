//! Skeleton models. The tables themselves are generated from the Python
//! definitions (`scripts/gen_skeletons.py` → `skeleton_tables.rs`); this
//! module adds the operations the app performs on them.

use std::collections::HashMap;

pub use crate::skeleton_tables::*;

/// One joint of a skeleton hierarchy, stored flat in pre-order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    pub name: &'static str,
    /// Keypoint index in the pose-estimation output (`None` for virtual joints).
    pub id: Option<u32>,
    /// Index of the parent node in `SkeletonModel::nodes`; `None` for the root.
    pub parent: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct SkeletonModel {
    pub name: &'static str,
    /// Root first, then descendants in pre-order (anytree's `descendants`).
    pub nodes: &'static [Node],
}

impl SkeletonModel {
    pub fn by_name(name: &str) -> Option<&'static SkeletonModel> {
        ALL_MODELS.iter().copied().find(|m| m.name == name)
    }

    pub fn root(&self) -> &Node {
        &self.nodes[0]
    }

    /// `(parent_name, child_name)` for every non-root node, in pre-order.
    pub fn pairs(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.nodes.iter().filter_map(|n| n.parent.map(|p| (self.nodes[p].name, n.name)))
    }

    /// `TRCViewer.update_skeleton_pairs`: pairs whose two markers both exist
    /// in `markers`, as marker indices, in pre-order.
    pub fn resolve_pairs(&self, markers: &[String]) -> Vec<(usize, usize)> {
        let index = |name: &str| markers.iter().position(|m| m == name);
        self.pairs().filter_map(|(a, b)| Some((index(a)?, index(b)?))).collect()
    }

    /// Keypoint id → joint name. When two nodes share an id the later one wins,
    /// matching the Python dict built in `DataManager.update_keypoint_names`.
    pub fn id_to_name(&self) -> HashMap<u32, &'static str> {
        self.nodes.iter().filter_map(|n| n.id.map(|id| (id, n.name))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_well_formed() {
        for m in ALL_MODELS {
            assert!(!m.nodes.is_empty(), "{}", m.name);
            assert_eq!(m.root().parent, None, "{}", m.name);
            for (i, n) in m.nodes.iter().enumerate().skip(1) {
                let p = n.parent.expect("non-root must have a parent");
                assert!(p < i, "{}: parent must precede child in pre-order", m.name);
            }
        }
        assert_eq!(APP_MODELS.len(), 11);
    }

    #[test]
    fn halpe26_pairs_match_python_traversal_order() {
        let pairs: Vec<_> = HALPE_26.pairs().take(4).collect();
        assert_eq!(pairs, vec![("Hip", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RBigToe")]);
    }

    #[test]
    fn resolve_pairs_skips_missing_markers() {
        let markers: Vec<String> = ["Hip", "RKnee", "RHip"].iter().map(|s| s.to_string()).collect();
        assert_eq!(HALPE_26.resolve_pairs(&markers), vec![(0, 2), (2, 1)]);
    }

    #[test]
    fn by_name_finds_every_table() {
        for m in ALL_MODELS {
            assert!(std::ptr::eq(SkeletonModel::by_name(m.name).unwrap(), *m));
        }
        assert!(SkeletonModel::by_name("nope").is_none());
    }
}
