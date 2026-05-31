import numpy as np
import cv2

class PlayerRatingSystem:
    def __init__(self):
        self.player_stats = {}

    def update_stats(self, tracks):
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, player_data in player_track.items():
                if player_id not in self.player_stats:
                    self.player_stats[player_id] = {
                        "speeds":        [],
                        "distances":     [],
                        "ball_contacts": 0,
                        "team":          None,
                        "frames_active": 0
                    }

                stats = self.player_stats[player_id]
                stats["frames_active"] += 1

                if "speed" in player_data:
                    stats["speeds"].append(player_data["speed"])

                if "distance" in player_data:
                    stats["distances"].append(player_data["distance"])

                if player_data.get("has_ball"):
                    stats["ball_contacts"] += 1

                if "team" in player_data:
                    stats["team"] = player_data["team"]

    def calculate_rating(self, player_id):
        if player_id not in self.player_stats:
            return 0

        stats = self.player_stats[player_id]

        speeds    = stats["speeds"]
        distances = stats["distances"]

        avg_speed      = np.mean(speeds) if speeds else 0
        max_speed      = max(speeds) if speeds else 0
        total_distance = max(distances) if distances else 0
        ball_contacts  = stats["ball_contacts"]
        frames_active  = stats["frames_active"]

        # Speed Score
        speed_score = min(10, (avg_speed / 30) * 10)

        # Distance Score
        distance_score = min(10, (total_distance / 500) * 10)

        # Ball Involvement Score
        involvement_rate  = (ball_contacts / frames_active * 100) if frames_active > 0 else 0
        involvement_score = min(10, (involvement_rate / 20) * 10)

        # Stamina Score
        if len(speeds) > 1:
            speed_variance = np.std(speeds)
            stamina_score  = max(0, 10 - (speed_variance / 5))
        else:
            stamina_score = 5

        # Work Rate Score
        work_rate       = avg_speed * total_distance
        work_rate_score = min(10, (work_rate / 5000) * 10)

        # Final Rating
        final_rating = (
            speed_score       * 0.25 +
            distance_score    * 0.25 +
            involvement_score * 0.20 +
            stamina_score     * 0.15 +
            work_rate_score   * 0.15
        )

        return round(final_rating, 1)

    def get_grade(self, rating):
        if rating >= 9:   return "S"
        elif rating >= 8: return "A"
        elif rating >= 7: return "B"
        elif rating >= 6: return "C"
        elif rating >= 5: return "D"
        else:             return "F"

    def get_all_ratings(self):
        ratings = {}

        # ← ADDED: Separate players by team
        team0_players = [
            (pid, stats) for pid, stats in self.player_stats.items()
            if stats["team"] == 0
        ]
        team1_players = [
            (pid, stats) for pid, stats in self.player_stats.items()
            if stats["team"] == 1
        ]

        # ← ADDED: Sort by frames_active and keep top 11 per team
        team0_players = sorted(
            team0_players,
            key=lambda x: x[1]["frames_active"],
            reverse=True
        )[:11]
        team1_players = sorted(
            team1_players,
            key=lambda x: x[1]["frames_active"],
            reverse=True
        )[:11]

        # ← ADDED: Combine selected players
        selected_players = team0_players + team1_players

        for player_id, stats in selected_players:
            rating    = self.calculate_rating(player_id)
            speeds    = stats["speeds"]
            distances = stats["distances"]

            ratings[str(player_id)] = {
                "player_id":      int(player_id),
                "rating":         float(rating),
                "team":           int(stats["team"]) if stats["team"] is not None else None,
                "avg_speed":      float(round(np.mean(speeds) if speeds else 0, 2)),
                "max_speed":      float(round(max(speeds) if speeds else 0, 2)),
                "total_distance": float(round(max(distances) if distances else 0, 2)),
                "ball_contacts":  int(stats["ball_contacts"]),
                "grade":          self.get_grade(rating)
            }
        return ratings