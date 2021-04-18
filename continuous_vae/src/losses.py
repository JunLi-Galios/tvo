import torch
from src import util
import numpy as np


def get_log_weight_log_p_log_q(generative_model, inference_network, obs, num_particles=1, reparam=True):
    """Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, obs_dim]
        num_particles: int

    Returns:
        log_weight: tensor of shape [batch_size, num_particles]
        log_p: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """
    latent_dist = inference_network.get_latent_dist(obs)
    latent = inference_network.sample_from_latent_dist(latent_dist, num_particles, reparam=reparam)

    log_p = generative_model.get_log_prob(latent, obs).transpose(0, 1)
    log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q
    return log_weight, log_p, log_q


def get_log_weight_and_log_q(generative_model, inference_network, obs, num_particles=1):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        log_weight: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """
    log_weight, log_p, log_q = get_log_weight_log_p_log_q(
        generative_model, inference_network, obs, num_particles=num_particles)
    return log_weight, log_q


def get_test_log_evidence(generative_model, inference_network, obs, num_particles=10):
    with torch.no_grad():
        log_weight, log_q = get_log_weight_and_log_q(
            generative_model, inference_network, obs, num_particles)
        log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
        iwae_log_evidence = torch.mean(log_evidence)
    return iwae_log_evidence


def get_iwae_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    log_weight, log_q = get_log_weight_and_log_q(generative_model, inference_network, obs, args.S)

    stable_log_weight = log_weight - torch.max(log_weight, 1)[0].unsqueeze(1)
    weight = torch.exp(stable_log_weight)
    normalized_weight = weight / torch.sum(weight, 1).unsqueeze(1)

    loss = -torch.mean(torch.sum(normalized_weight.detach() * log_weight, 1), 0)

    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return loss, iwae_estimate


def get_vae_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    log_weight, log_q = get_log_weight_and_log_q(generative_model, inference_network, obs, args.S)
    train_elbo = torch.mean(log_weight)
    loss = -train_elbo

    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return loss, iwae_estimate


def get_reinforce_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(generative_model, inference_network, obs, args.S)
    reinforce = log_weight.detach() * log_q + log_weight
    loss = -torch.mean(reinforce)

    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return loss, iwae_estimate


def get_thermo_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_p, log_q = get_log_weight_log_p_log_q(
        generative_model, inference_network, obs, num_particles=args.S, reparam=False)
    thermo_loss = get_thermo_loss_from_log_weight_log_p_log_q(
        log_weight, log_p, log_q, args.partition, num_particles=args.S, integration=args.integration)
    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return thermo_loss, iwae_estimate


def get_thermo_alpha_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_p, log_q = get_log_weight_log_p_log_q(
        generative_model, inference_network, obs, num_particles=args.S, reparam=False)
    thermo_alpha_loss = get_thermo_alpha_loss_from_log_weight_log_p_log_q(
        args.alpha, log_weight, log_p, log_q, args.partition, num_particles=args.S, integration=args.integration)
    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return thermo_alpha_loss, iwae_estimate


def get_thermo_loss_from_log_weight_log_p_log_q(log_weight, log_p, log_q, partition, num_particles=1,
                                                integration='left'):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]
        log_p: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """


    heated_log_weight = log_weight.unsqueeze(-1) * partition
    heated_normalized_weight = util.exponentiate_and_normalize(
        heated_log_weight, dim=1)
    thermo_logp = partition * log_p.unsqueeze(-1) + \
        (1 - partition) * log_q.unsqueeze(-1)

    wf = heated_normalized_weight * log_weight.unsqueeze(-1)
    w_detached = heated_normalized_weight.detach()
    wf_detached = wf.detach()
    if num_particles == 1:
        correction = 1
    else:
        correction = num_particles / (num_particles - 1)

    cov = correction * torch.sum(
        w_detached * (log_weight.unsqueeze(-1) - torch.sum(wf, dim=1, keepdim=True)).detach() *
        (thermo_logp - torch.sum(thermo_logp * w_detached, dim=1, keepdim=True)),
        dim=1)

    multiplier = torch.zeros_like(partition)
    if integration == 'trapz':
        multiplier[0] = 0.5 * (partition[1] - partition[0])
        multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
        multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
    elif integration == 'left':
        multiplier[:-1] = partition[1:] - partition[:-1]
    elif integration == 'right':
        multiplier[1:] = partition[1:] - partition[:-1]

    loss = -torch.mean(torch.sum(
        multiplier * (cov + torch.sum(
            w_detached * log_weight.unsqueeze(-1), dim=1)),
        dim=1))

    return loss


# def get_thermo_alpha_loss_from_log_weight_log_p_log_q(alpha, log_weight, log_p, log_q, partition, num_particles=1,
#                                                 integration='left'):
#     """Args:
#         log_weight: tensor of shape [batch_size, num_particles]
#         log_p: tensor of shape [batch_size, num_particles]
#         log_q: tensor of shape [batch_size, num_particles]
#         partition: partition of [0, 1];
#             tensor of shape [num_partitions + 1] where partition[0] is zero and
#             partition[-1] is one;
#             see https://en.wikipedia.org/wiki/Partition_of_an_interval
#         num_particles: int
#         integration: left, right or trapz

#     Returns:
#         loss: scalar that we call .backward() on and step the optimizer.
#         elbo: average elbo over data
#     """
#     print('---------------------new iteration-----------------')
    
#     heated_log_pi = util.alpha_average(log_p.unsqueeze(-1), log_q.unsqueeze(-1), partition, alpha)
#     heated_log_p = partition * log_p.unsqueeze(-1)
#     heated_log_q = partition * log_q.unsqueeze(-1)
#     log_heated_normalized_w = util.lognormexp(
#         heated_log_pi - heated_log_q, dim=1)
    
#     log_w_detached = log_heated_normalized_w.detach()
#     w_detached = torch.exp(log_w_detached)
    
# #     print('log_w_detached', log_w_detached.min(), log_w_detached.max())
    
#     heated_log_f_L = (heated_log_pi - heated_log_p) * (alpha -1)
#     heated_log_f_R = (heated_log_pi - heated_log_q) * (alpha -1)
    
#     m1 = heated_log_f_L - heated_log_f_R
#     print('heated_log_f_L - heated_log_f_R', m1.min(), m1.max())
    
# #     print('heated_log_f_L', heated_log_f_L.min(), heated_log_f_L.max())
# #     print('heated_log_f_R', heated_log_f_R.min(), heated_log_f_R.max())
    
#     heated_f_L = torch.exp(heated_log_f_L)
#     heated_f_R = torch.exp(heated_log_f_R)
    
#     heated_log_L = log_w_detached + heated_log_f_L
    
#     heated_log_R = log_w_detached + heated_log_f_R
    
# #     L = torch.exp(torch.logsumexp(
# #         torch.exp(log_w_detached) * heated_log_f_L, dim=1))
# #     R = torch.exp(torch.logsumexp(
# #         torch.exp(log_w_detached) * heated_log_f_R, dim=1))

    
#     log_wf_L_detached = heated_log_L.detach()
#     log_wf_R_detached = heated_log_R.detach()
    
    
#     if num_particles == 1:
#         correction = 1
#     else:
#         correction = num_particles / (num_particles - 1)
        
#     cov_L = correction * torch.sum(
#         w_detached * (heated_f_L - torch.sum(heated_f_L, dim=1, keepdim=True)).detach() *
#         (heated_log_pi - torch.sum(heated_log_pi * w_detached, dim=1, keepdim=True)),
#         dim=1)
#     cov_R = correction * torch.sum(
#         w_detached * (heated_f_R - torch.sum(heated_f_R, dim=1, keepdim=True)).detach() *
#         (heated_log_pi - torch.sum(heated_log_pi * w_detached, dim=1, keepdim=True)),
#         dim=1)
        
#     multiplier = torch.zeros_like(partition)
#     if integration == 'trapz':
#         multiplier[0] = 0.5 * (partition[1] - partition[0])
#         multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
#         multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
#     elif integration == 'left':
#         multiplier[:-1] = partition[1:] - partition[:-1]
#     elif integration == 'right':
#         multiplier[1:] = partition[1:] - partition[:-1]
        
#     L = torch.sum(multiplier * ( torch.exp(torch.logsumexp(
#         heated_log_f_L, dim=1))), dim=1)
    
#     R = torch.sum(multiplier * ( torch.exp(torch.logsumexp(
#         heated_log_f_R, dim=1))), dim=1)
# #     R = 0

# #     print('L={}, R={}'.format(L,R))
#     print('log_p-log_q', (log_p-log_q).min(), (log_p-log_q).max())
#     print('L-R', (L-R).min(), (L-R).max())
    
# #     L = torch.sum(multiplier * (cov_L + torch.exp(torch.logsumexp(
# #         heated_log_L, dim=1))), dim=1)
# #     R = torch.sum(multiplier * (cov_R + torch.exp(torch.logsumexp(
# #         heated_log_R, dim=1))), dim=1)
    
#     loss = -torch.mean(L-R) / (1-alpha)
    

#     heated_log_weight = log_weight.unsqueeze(-1) * partition
    
#     # log(p/q)
#     log_alpha_weight = util.log_alpha(torch.exp(log_weight.unsqueeze(-1)), alpha)
#     # log(p/q) * q
#     log_alpha_weight_q = log_alpha_weight * torch.exp(log_q.unsqueeze(-1))
#     # beta*log(p/q)
#     heated_log_alpha_weight = log_alpha_weight * partition
#     # exp[beta*log(p/q)]
#     heated_exp_beta_weight = util.exp_alpha(heated_log_alpha_weight, alpha)
#     # pi_beta = exp[beta*log(p/q)] * q
#     pi_beta = heated_exp_beta_weight * torch.exp(log_q.unsqueeze(-1))
#     # pi_beta^{alpha}
#     pi_beta_power = torch.pow(pi_beta, alpha)
    
#     normalization_z = torch.mean(pi_beta_power * log_alpha_weight, dim=1).detach()
    
#     pi_beta_power_weight = torch.div(pi_beta_power, torch.exp(log_q.unsqueeze(-1)))
#     pi_beta_power_weight_detach = pi_beta_power_weight.detach()
#     loss_1 = -torch.mean(pi_beta_power_weight_detach * log_alpha_weight_q, dim=1)
    
#     log_alpha_weight_detach = log_alpha_weight.detach()
#     loss_2 = -torch.mean(pi_beta_power * log_alpha_weight_detach, dim=1)

#     multiplier = torch.zeros_like(partition)
#     if integration == 'trapz':
#         multiplier[0] = 0.5 * (partition[1] - partition[0])
#         multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
#         multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
#     elif integration == 'left':
#         multiplier[:-1] = partition[1:] - partition[:-1]
#     elif integration == 'right':
#         multiplier[1:] = partition[1:] - partition[:-1]

#     loss = torch.sum(multiplier * (loss_1 + loss_2))
    
#     normalization = torch.sum(multiplier * normalization_z)
    

#     return loss


def get_thermo_alpha_loss_from_log_weight_log_p_log_q(alpha, log_weight, log_p, log_q, partition, num_particles=1,
                                                integration='left'):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]
        log_p: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
#     print('---------------------new iteration-----------------')

    multiplier = torch.zeros_like(partition)
    if integration == 'trapz':
        multiplier[0] = 0.5 * (partition[1] - partition[0])
        multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
        multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
    elif integration == 'left':
        multiplier[:-1] = partition[1:] - partition[:-1]
    elif integration == 'right':
        multiplier[1:] = partition[1:] - partition[:-1]
    
    heated_log_pi = util.alpha_average(log_p.unsqueeze(-1), log_q.unsqueeze(-1), partition, alpha)
    heated_log_p = partition * log_p.unsqueeze(-1)
    heated_log_q = partition * log_q.unsqueeze(-1)
    
    heated_log_w1_L = np.log(alpha) + (alpha - 1) * (heated_log_pi - heated_log_p) - heated_log_q
    heated_log_w1_R = np.log(alpha) + (alpha - 1) * (heated_log_pi - heated_log_q) - heated_log_q
    heated_log_w2_L = np.log(1 - alpha) + (alpha) * (heated_log_pi - heated_log_p) - heated_log_q
    heated_log_w2_R = np.log(1 - alpha) + (alpha) * (heated_log_pi - heated_log_q) - heated_log_q
    
    heated_log_w1_L_detach = heated_log_w1_L.detach()
    heated_log_w1_R_detach = heated_log_w1_R.detach()
    heated_log_w2_L_detach = heated_log_w2_L.detach()
    heated_log_w2_R_detach = heated_log_w2_R.detach()
    
    heated_log_L1 = heated_log_w1_L_detach + heated_log_pi
    heated_log_L2 = heated_log_w2_L_detach + heated_log_p
    heated_log_R1 = heated_log_w1_R_detach + heated_log_pi
    heated_log_R2 = heated_log_w2_R_detach + heated_log_q

    
    thermo_log_L1 = torch.logsumexp(torch.log(multiplier) + torch.logsumexp(heated_log_L1, dim=1),dim=1)
    thermo_log_L2 = torch.logsumexp(torch.log(multiplier) + torch.logsumexp(heated_log_L2, dim=1),dim=1)
    thermo_log_R1 = torch.logsumexp(torch.log(multiplier) + torch.logsumexp(heated_log_R1, dim=1),dim=1)
    thermo_log_R2 = torch.logsumexp(torch.log(multiplier) + torch.logsumexp(heated_log_R2, dim=1),dim=1)
    
    diff1 = thermo_log_L1 - thermo_log_R2
    diff2 = thermo_log_L2 - thermo_log_R2
    diff3 = thermo_log_R1 - thermo_log_R2
    diff4 = thermo_log_R2 - thermo_log_R2
    
#     print('thermo_log_L1', thermo_log_L1.size(), thermo_log_L1.min(), thermo_log_L1.max())
#     print('thermo_log_L2', thermo_log_L2.size(), thermo_log_L2.min(), thermo_log_L2.max())
#     print('thermo_log_R1', thermo_log_R1.size(), thermo_log_R1.min(), thermo_log_R1.max())
#     print('thermo_log_R2', thermo_log_R2.size(), thermo_log_R2.min(), thermo_log_R2.max())
    
    denominator = torch.exp(diff1) + torch.exp(diff2) - torch.exp(diff3) - torch.exp(diff4)
    denominator_detach = denominator.detach()
    
#     print('denominator', denominator.size(), denominator.min(), denominator.max())
    
    loss = -torch.div(denominator, denominator_detach + 1e-10)
    
#     print('loss', loss.size(), loss.min(), loss.max())
    
    loss = torch.mean(loss) / (1-alpha)
    
    return loss


def get_log_p_and_kl(generative_model, inference_network, obs, num_samples):
    """Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_samples: int

    Returns:
        log_p: tensor of shape [batch_size]
        kl: tensor of shape [batch_size]
    """

    log_weight, _ = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_samples)
    log_p = torch.logsumexp(log_weight, dim=1) - np.log(num_samples)
    elbo = torch.mean(log_weight, dim=1)
    kl = log_p - elbo
    return log_p, kl


def get_log_p_and_alpha_div(generative_model, inference_network, obs, num_samples):
    """Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_samples: int
    Returns:
        log_p: tensor of shape [batch_size]
        kl: tensor of shape [batch_size]
    """

    log_weight, log_p, log_q = get_log_weight_log_p_log_q(
        generative_model, inference_network, obs, num_samples)
    
    # log(p/q)
    log_alpha_weight = util.log_alpha(torch.exp(log_weight.unsqueeze(-1)), alpha)
    
    log_p = torch.logsumexp(log_weight, dim=1) - np.log(num_samples)
    beta_0 = torch.mean(log_alpha_weight, dim=1)
    return log_p, beta_0
