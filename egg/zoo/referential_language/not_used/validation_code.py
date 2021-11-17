print("| STARTING TEST")

_, test_interaction = trainer.eval(validation_loader)

dump = dict((k, v.mean().item()) for k, v in test_interaction.aux.items())
dump.update(dict((k, v.mean().item()) for k, v in test_interaction.aux_input.items()))
dump.update(dict(mode="VALIDATION"))
print(json.dumps(dump), flush=True)

if opts.wandb and opts.distributed_context.is_leader:
    wandb.log(
        {"post_hoc_validation_accuracy": test_interaction.aux["acc"].mean().item()},
        commit=True,
    )
