from rich.table import Table
from rich.console import Console
from rich.style import Style


console = Console()

def print_loss_table(epoch, total_loss, mse_pde_1, mse_pde_2, mse_pde_3, mse_obs_1, mse_obs_2):
    table = Table(title=f"Epoch {epoch} Losses")
    table.add_column("Loss Type")
    table.add_column("Loss Value")
    table.add_row("Total Loss", f"{total_loss.item():.2e}")
    table.add_row("PDE 1 Loss", f"{mse_pde_1.item():.2e}")
    table.add_row("PDE 2 Loss", f"{mse_pde_2.item():.2e}")
    table.add_row("PDE 3 Loss", f"{mse_pde_3.item():.2e}")
    table.add_row("Obs 1 Loss", f"{mse_obs_1.item():.2e}")
    table.add_row("Obs 2 Loss", f"{mse_obs_2.item():.2e}")
    console.print(table)

def print_epoch_err(epoch, err):
    epoch_style = Style(color="cyan")
    err_style = Style(color="red")
    console.print(f"[{epoch_style}]Epoch:[/] [{err_style}]{epoch}[/] [{epoch_style}] l2 relative err:[/] [{err_style}]{err}[/]")
