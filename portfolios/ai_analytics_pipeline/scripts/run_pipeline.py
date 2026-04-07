from campaign_eval.config import CampaignConfig
from campaign_eval.pipeline import CampaignEvaluationPipeline


if __name__ == "__main__":
    pipeline = CampaignEvaluationPipeline(CampaignConfig())
    results = pipeline.run_from_csv("data/mock")
    pipeline.save_artifacts(results)
    print(results["summary"]["business_readout"])
