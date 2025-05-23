def train_cross_domain_transfer_agent(model_class, model_name, env: MultiAgentEnv, difficulty, train_parameters={}):
    # Load expert model from domain source
    expert_model_path = train_parameters.get("expert_model_path")
    assert expert_model_path and os.path.exists(expert_model_path), "Expert model path is required."

    # Create MARWIL config for imitation transfer
    config = MARWILConfig().environment(env=env).framework("torch")
    config.rollouts(num_rollout_workers=0)
    config.training(train_batch_size=2000)

    # Load expert model and use its policy to generate demonstration data
    expert_model: Algorithm = model_class.load(expert_model_path)
    expert_policy: Policy = expert_model.get_policy()

    # Collect demonstrations from expert
    rollout_worker = RolloutWorker(env_creator=lambda _: env, policy=expert_policy)
    demo_batches = []
    for _ in range(10):
        sample_batch = rollout_worker.sample()
        demo_batches.append(sample_batch)

    # Combine all batches
    combined_batch = SampleBatch.concat_samples(demo_batches)

    # Train the MARWIL agent on the demonstration data
    imitation_agent: Algorithm = config.build()
    imitation_agent.train_on_batch(combined_batch)

    # Save the agent for evaluation/transfer
    model_path = Path("models") / f"{model_name}_{difficulty}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    imitation_agent.save(str(model_path))

    return CrossDomainTransfer(imitation_agent), combined_batch["rewards"]
