
import json
import requests
from analytics.settings import settings
from analytics.utils import display_runtime


@display_runtime
def pull_reviews(game_code):
    with requests.Session() as req:

        print(f"Getting {game_code} reviews...")

        num_per_page = 100
        num_reviews = 100
        cursor = "*"
        page_no = 1
        log_directory = settings.log_directory

        while num_reviews == 100:

            params = {
                "filter": "recent", 
                "cursor": cursor, 
                "num_per_page": num_per_page
            }
            response = req.get(
                url=f"https://store.steampowered.com/appreviews/{game_code}?json=1",
                params=params
                )
            response_json = response.json()

            if response.status_code == 200:

                if cursor == "*":
                    with open(f"{log_directory}\\{game_code}-summary.json", "w", encoding="utf-8") as f:
                        f.write(json.dumps(
                            {"summary": response_json.get("query_summary")},
                            indent=4))

                num_reviews = response_json.get("query_summary").get("num_reviews")
                cursor = response_json.get("cursor")

                with open(f"{log_directory}\\{game_code}-{page_no}.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps({"reviews": response_json.get("reviews")}, indent=4))

            page_no += 1
