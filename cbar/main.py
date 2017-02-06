import click
from .cross_validation import cv


@click.group()
def cli():
    pass


@cli.group()
@click.option('--dataset', '-d',
              type=click.Choice(['freesound', 'cal500', 'cal10k']),
              help='Which dataset to evaluate the retrieval method on')
@click.option('--codebook-size', '-c',
              type=click.Choice([512, 1024, 2048, 4096]),
              default=512,
              help='Codebook size for encoding the dataset')
@click.option('--n-folds', '-k', default=5,
              help='Number of folds (default: 5)')
@click.pass_context
def crossval(ctx, dataset, codebook_size, n_folds):
    ctx.obj = dict(dataset=dataset,
                   codebook_size=codebook_size,
                   n_folds=n_folds)


@crossval.command()
@click.pass_context
@click.option('--max-iter', '-n', default=100000,
              help='Maximum number of iterations')
@click.option('--valid-interval', '-i', default=10000,
              help='Valiate model every i iterations')
@click.option('-k', default=30, help='Rank of parameter matrix W')
@click.option('--n0', default=1.0, help='Step size parameter 1')
@click.option('--n1', default=0.0, help='Step size parameter 2')
@click.option('--rank-thresh', '-t', default=0.1,
              help='Threshold for early stopping')
@click.option('--lambda', '-l', 'lambda_', default=0.1,
              help='Regularization constant')
@click.option('--loss', type=click.Choice(['warp', 'auc']), default='warp',
              help='Loss function')
@click.option('--max-dips', '-d', default=10,
              help='Maximum number of dips')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbosity')
def loreta(ctx, **kwargs):
    cv(ctx.obj['dataset'], ctx.obj['codebook_size'], 'loreta', **kwargs)


@crossval.command()
@click.pass_context
@click.option('--max-iter', '-n', default=100000,
              help='Maximum number of iterations')
@click.option('--valid-interval', '-i', default=10000,
              help='Loss function')
@click.option('--max-dips', '-d', default=10,
              help='Maximum number of dips')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbosity')
def pamir(ctx, **kwargs):
    cv(ctx.obj['dataset'], ctx.obj['codebook_size'], 'pamir', **kwargs)


@crossval.command()
@click.pass_context
@click.option('--n_estimators', '-n', default=100,
              help='Number of trees to build in each forest')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbosity')
def random_forest(ctx, **kwargs):
    cv(ctx.obj['dataset'], ctx.obj['codebook_size'], 'random-forest', **kwargs)
