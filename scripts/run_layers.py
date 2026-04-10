from scripts.run import *

@hydra.main(config_path="../config", config_name="config_layers", version_base=None)
def main(cfg=None):
    set_seed(cfg.seed)
    mode = cfg.wandb.mode
    if cfg.load_state:
        logger.info(f"Loading config from {cfg.config_path}")
        engine = load_model_instance(cfg)
        cfg = engine._config
        os.environ["WANDB_DIR"] = cfg.config_path.split("wandb")[0]
        iden = get_project_id(cfg.run_path)
        # wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode,
        #         resume='allow', id=iden)
        wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode)
        # Log metrics with wandb
        wandb.watch(engine.model)
    else:
        engine = setup_model(config=cfg)
        if not is_distributed() or is_master():
            wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode)
            wandb.watch(engine.model)

    engine.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        engine.optimiser,
        T_max=cfg.engine.lr_scheduler_T_max,
        eta_min=cfg.engine.lr_scheduler_eta_min,
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

    run(engine, callback)


if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")
