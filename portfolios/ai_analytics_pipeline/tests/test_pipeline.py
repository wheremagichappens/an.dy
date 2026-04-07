from campaign_eval.config import CampaignConfig
from campaign_eval.data_generation import generate_mock_data
from campaign_eval.pipeline import CampaignEvaluationPipeline


def test_pipeline_end_to_end() -> None:
    data = generate_mock_data(n_users=500, random_state=7)
    pipeline = CampaignEvaluationPipeline(CampaignConfig(caliper=0.08))
    results = pipeline.run(
        users_df=data["users"],
        trips_df=data["trips"],
        treatment_df=data["treatment"],
        weekly_df=data["weekly_panel"],
    )

    summary = results["summary"]
    assert summary["matched_pairs"] > 0
    assert "business_readout" in summary
    assert results["balance"].shape[0] > 0
