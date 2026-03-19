use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Equal,
    NotEqual,
    And,
    Or,
    AddDot,
    SubDot,
    MulDot,
    DivDot,
    LessDot,
    GreaterDot,
    LessEqualDot,
    GreaterEqualDot,
    EqualDot,
    NotEqualDot,
    AndDot,
    OrDot,
    XorDot,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Subtract => write!(f, "-"),
            BinaryOp::Multiply => write!(f, "*"),
            BinaryOp::Divide => write!(f, "/"),
            BinaryOp::Modulo => write!(f, "%"),
            BinaryOp::Less => write!(f, "<"),
            BinaryOp::Greater => write!(f, ">"),
            BinaryOp::LessEqual => write!(f, "<="),
            BinaryOp::GreaterEqual => write!(f, ">="),
            BinaryOp::Equal => write!(f, "=="),
            BinaryOp::NotEqual => write!(f, "!="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
            BinaryOp::AddDot => write!(f, ".+"),
            BinaryOp::SubDot => write!(f, ".-"),
            BinaryOp::MulDot => write!(f, ".*"),
            BinaryOp::DivDot => write!(f, "./"),
            BinaryOp::LessDot => write!(f, ".<"),
            BinaryOp::GreaterDot => write!(f, ".>"),
            BinaryOp::LessEqualDot => write!(f, ".<="),
            BinaryOp::GreaterEqualDot => write!(f, ".>="),
            BinaryOp::EqualDot => write!(f, ".=="),
            BinaryOp::NotEqualDot => write!(f, ".!="),
            BinaryOp::AndDot => write!(f, ".&"),
            BinaryOp::OrDot => write!(f, ".|"),
            BinaryOp::XorDot => write!(f, ".^"),
        }
    }
}
