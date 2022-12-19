use crate::splitter::NodeInfo;

#[derive(Debug)]
pub enum MissingInfo {
    Left,
    Right,
    Branch(NodeInfo),
}

pub trait MissingHandler {
    fn handle_missing() -> MissingInfo;
}
