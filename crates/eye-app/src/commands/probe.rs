use crate::ctx::Ctx;

#[derive(Debug, clap::Args)]
pub struct Args {}

pub fn run(_ctx: &Ctx, _args: Args) -> anyhow::Result<()> {
    anyhow::bail!("`eye probe` is not implemented yet")
}
