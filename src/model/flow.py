 # Normalizing Flow
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

def build_flow(param_dim, context_dim, hidden_dim, n_transforms):
    transforms = []
    for _ in range(n_transforms):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=param_dim,
                hidden_features=hidden_dim,
                context_features=context_dim,
            )
        )

    transform = CompositeTransform(transforms)
    base = StandardNormal([param_dim])
    return Flow(transform, base)
