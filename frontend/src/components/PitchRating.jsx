import { useState, useEffect } from "react";
import axios from "axios";
import { API_URL } from "../api";

const PitchRating = ({ videoFilename }) => {
  const [ratings, setRatings] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    if (videoFilename) fetchRatings();
  }, [videoFilename]);

  const fetchRatings = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(
        `${API_URL}/ratings/${videoFilename}`
      );
      setRatings(response.data);
    } catch (err) {
      setError("Ratings not available yet.");
    } finally {
      setLoading(false);
    }
  };

  const getRatingColor = (rating) => {
    if (rating >= 8) return "#22c55e";
    if (rating >= 6) return "#eab308";
    if (rating >= 4) return "#f97316";
    return "#ef4444";
  };

  const getTextColor = (rating) => {
    if (rating >= 8) return "#000";
    return "#fff";
  };

  if (loading) return (
    <div className="text-center text-p4 py-10">
      Loading player ratings...
    </div>
  );

  if (error) return (
    <div className="text-center text-red-400 py-10">{error}</div>
  );

  if (!ratings) return null;

  const teamA = Object.values(ratings).filter(p => p.team === 0);
  const teamB = Object.values(ratings).filter(p => p.team === 1);

  const getPositions = (players, side) => {
    return players.map((player, index) => {
      const total = players.length;
      const yPercent = ((index + 1) / (total + 1)) * 100;
      const xPercent = side === "left" ? 25 : 75;
      return { ...player, x: xPercent, y: yPercent };
    });
  };

  const teamAPositioned = getPositions(teamA, "left");
  const teamBPositioned = getPositions(teamB, "right");
  const allPlayers = [...teamAPositioned, ...teamBPositioned];

  return (
    <div className="bg-s2 rounded-2xl p-6 mt-4">

      {/* Header */}
      <h2 className="h5 text-p4 mb-2 text-center">
        ⭐ Player Ratings
      </h2>
      <p className="text-center text-p5 text-sm mb-6">
        Click on a player to see detailed stats
      </p>

      {/* Pitch */}
      <div
        className="relative mx-auto rounded-xl overflow-hidden"
        style={{
          width: "100%",
          maxWidth: "800px",
          aspectRatio: "16/10",
          background: "linear-gradient(180deg, #166534 0%, #15803d 50%, #166534 100%)",
          border: "3px solid #fff",
        }}
      >
        {/* Pitch Lines */}
        <svg
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 800 500"
          preserveAspectRatio="none"
        >
          <line x1="400" y1="0" x2="400" y2="500"
            stroke="rgba(255,255,255,0.5)" strokeWidth="2" />
          <circle cx="400" cy="250" r="70"
            fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" />
          <circle cx="400" cy="250" r="4"
            fill="rgba(255,255,255,0.5)" />
          <rect x="0" y="150" width="120" height="200"
            fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" />
          <rect x="0" y="200" width="50" height="100"
            fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" />
          <rect x="680" y="150" width="120" height="200"
            fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" />
          <rect x="750" y="200" width="50" height="100"
            fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" />
          <rect x="2" y="2" width="796" height="496"
            fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="2" />
        </svg>

        {/* Team Labels */}
        <div className="absolute top-2 left-4 text-white text-xs font-bold opacity-70">
          TEAM A
        </div>
        <div className="absolute top-2 right-4 text-white text-xs font-bold opacity-70">
          TEAM B
        </div>

        {/* Players */}
        {allPlayers.map((player) => (
          <div
            key={player.player_id}
            onClick={() => setSelected(
              selected?.player_id === player.player_id ? null : player
            )}
            style={{
              position:  "absolute",
              left:      `${player.x}%`,
              top:       `${player.y}%`,
              transform: "translate(-50%, -50%)",
              cursor:    "pointer",
              zIndex:    10,
            }}
          >
            <div
              style={{
                width:           "48px",
                height:          "48px",
                borderRadius:    "50%",
                background:      getRatingColor(player.rating),
                border:          selected?.player_id === player.player_id
                                   ? "3px solid #fff"
                                   : "2px solid rgba(255,255,255,0.5)",
                display:         "flex",
                flexDirection:   "column",
                alignItems:      "center",
                justifyContent:  "center",
                boxShadow:       "0 2px 8px rgba(0,0,0,0.5)",
                transition:      "all 0.2s",
                transform:       selected?.player_id === player.player_id
                                   ? "scale(1.2)"
                                   : "scale(1)",
              }}
            >
              <span style={{
                fontSize:   "11px",
                fontWeight: "900",
                color:      getTextColor(player.rating),
                lineHeight: "1",
              }}>
                {player.rating}
              </span>
              <span style={{
                fontSize:   "8px",
                fontWeight: "700",
                color:      getTextColor(player.rating),
                lineHeight: "1",
              }}>
                ID:{player.player_id}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Rating Legend */}
      <div className="flex justify-center gap-6 mt-4 flex-wrap">
        {[
          { color: "#22c55e", label: "8+ Excellent" },
          { color: "#eab308", label: "6-8 Good" },
          { color: "#f97316", label: "4-6 Average" },
          { color: "#ef4444", label: "0-4 Poor" },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div style={{
              width: "12px", height: "12px",
              borderRadius: "50%", background: color
            }} />
            <span className="text-p5 text-xs">{label}</span>
          </div>
        ))}
      </div>

      {/* Selected Player Stats */}
      {selected && (
        <div className="mt-6 bg-s1 rounded-xl p-5 border border-s3 max-w-md mx-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-p4 font-bold text-lg">
              Player {selected.player_id}
            </h3>
            <div className="flex items-center gap-2">
              <span
                className="text-2xl font-black"
                style={{ color: getRatingColor(selected.rating) }}
              >
                {selected.grade}
              </span>
              <span className="text-p4 font-bold text-xl">
                {selected.rating}/10
              </span>
            </div>
          </div>

          {/* Rating Bar */}
          <div className="w-full bg-s3 rounded-full h-3 mb-4">
            <div
              className="h-3 rounded-full transition-all"
              style={{
                width:      `${(selected.rating / 10) * 100}%`,
                background: getRatingColor(selected.rating)
              }}
            />
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "⚡ Avg Speed",     value: `${selected.avg_speed} km/h` },
              { label: "🏃 Max Speed",     value: `${selected.max_speed} km/h` },
              { label: "📏 Distance",      value: `${selected.total_distance} m` },
              { label: "⚽ Ball Contacts", value: selected.ball_contacts },
              { label: "👕 Team",          value: selected.team === 0 ? "Team A" : "Team B" },
              { label: "🏅 Grade",         value: selected.grade },
            ].map(({ label, value }) => (
              <div key={label} className="bg-s2 rounded-lg p-3">
                <p className="text-p5 text-xs mb-1">{label}</p>
                <p className="text-p4 font-bold">{value}</p>
              </div>
            ))}
          </div>

          <button
            onClick={() => setSelected(null)}
            className="mt-4 w-full py-2 bg-s3 text-p5 rounded-lg text-sm hover:bg-s4 transition-colors"
          >
            Close
          </button>
        </div>
      )}

      {/* Team Summary */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        {[
          { team: 0, label: "Team A", players: teamA },
          { team: 1, label: "Team B", players: teamB },
        ].map(({ team, label, players }) => {
          const avgRating = players.length
            ? (players.reduce((s, p) => s + p.rating, 0) / players.length).toFixed(1)
            : 0;
          const bestPlayer = players.reduce(
            (best, p) => p.rating > (best?.rating || 0) ? p : best, null
          );
          return (
            <div key={team} className="bg-s1 rounded-xl p-4 border border-s3">
              <h3 className="text-p4 font-bold mb-3">{label}</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-p5 text-sm">Players</span>
                  <span className="text-p4 font-bold">{players.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-p5 text-sm">Avg Rating</span>
                  <span
                    className="font-bold"
                    style={{ color: getRatingColor(parseFloat(avgRating)) }}
                  >
                    {avgRating}/10
                  </span>
                </div>
                {bestPlayer && (
                  <div className="flex justify-between">
                    <span className="text-p5 text-sm">Best Player</span>
                    <span className="text-p4 font-bold">
                      ID:{bestPlayer.player_id} ({bestPlayer.rating})
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
};

export default PitchRating;