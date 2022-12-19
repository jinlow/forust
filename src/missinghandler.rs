use crate::splitter::NodeInfo;

#[derive(Debug)]
pub enum MissingInfo {
    Left,
    Right,
    Branch(NodeInfo),
}
