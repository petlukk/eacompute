use crate::ast::Stmt;
use crate::error::CompileError;
use crate::lexer::{Span, TokenKind};

use super::Parser;

impl Parser {
    pub(super) fn parse_foreach(&mut self) -> crate::error::Result<Stmt> {
        let start = self.current_position();
        self.advance(); // consume 'foreach'
        self.expect_kind(TokenKind::LeftParen, "expected '(' after 'foreach'")?;
        let var_token = self.expect_kind(TokenKind::Identifier, "expected loop variable name")?;
        let var = var_token.lexeme.clone();
        self.expect_kind(TokenKind::In, "expected 'in' after loop variable")?;
        let start_expr = self.expression()?;
        self.expect_kind(TokenKind::DotDot, "expected '..' in range")?;
        let end_expr = self.expression()?;
        self.expect_kind(TokenKind::RightParen, "expected ')' after range")?;
        self.expect_kind(TokenKind::LeftBrace, "expected '{' after foreach header")?;
        let body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after foreach body")?;
        let end = self.previous_position();
        Ok(Stmt::ForEach {
            var,
            start: start_expr,
            end: end_expr,
            body,
            span: Span::new(start, end),
        })
    }

    pub(super) fn parse_while(&mut self) -> crate::error::Result<Stmt> {
        let start = self.current_position();
        self.advance(); // consume 'while'
        let condition = self.expression()?;
        self.expect_kind(TokenKind::LeftBrace, "expected '{' after while condition")?;
        let body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after while body")?;
        let end = self.previous_position();
        Ok(Stmt::While {
            condition,
            body,
            span: Span::new(start, end),
        })
    }

    pub(super) fn parse_for_loop(&mut self) -> crate::error::Result<Stmt> {
        let start_pos = self.current_position();
        self.advance(); // consume 'for'
        let var_token =
            self.expect_kind(TokenKind::Identifier, "expected loop variable after 'for'")?;
        let var = var_token.lexeme.clone();
        self.expect_kind(TokenKind::In, "expected 'in' after loop variable")?;
        let start_expr = self.expression()?;
        self.expect_kind(TokenKind::DotDot, "expected '..' in range")?;
        let end_expr = self.expression()?;
        let step = if self.check_identifier("step") {
            self.advance();
            let step_tok =
                self.expect_kind(TokenKind::IntLiteral, "expected integer after 'step'")?;
            Some(step_tok.lexeme.parse::<u32>().map_err(|_| {
                CompileError::parse_error(
                    "step must be a positive integer",
                    step_tok.position.clone(),
                )
            })?)
        } else {
            None
        };
        self.expect_kind(TokenKind::LeftBrace, "expected '{' after for header")?;
        let body = self.parse_block()?;
        self.expect_kind(TokenKind::RightBrace, "expected '}' after for body")?;
        let end_pos = self.previous_position();
        Ok(Stmt::For {
            var,
            start: start_expr,
            end: end_expr,
            step,
            body,
            span: Span::new(start_pos, end_pos),
        })
    }
}
