from campaign_eval.data_generation import save_mock_data


if __name__ == "__main__":
    save_mock_data("data/mock", n_users=1500, random_state=42)
    print("Mock data written to data/mock")
