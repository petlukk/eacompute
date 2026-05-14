pub mod error;
pub mod manifest;
pub mod report;

pub use error::BenchError;

#[cfg(feature = "llvm")]
pub fn run(_args: &[String]) -> Result<i32, BenchError> {
    Err(BenchError::Unsupported(
        "ea bench: implementation pending (Task 6 of the plan)".to_string(),
    ))
}
