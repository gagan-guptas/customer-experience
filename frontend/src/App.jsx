import { useState, useEffect } from "react";
import { Bar, Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

// Helper function to categorize sentiment
const getSentimentCategory = (score) => {
  if (score === 1) return "Positive";
  if (score === 0) return "Neutral";
  if (score === -1) return "Negative";
  return "Unknown"; // Should ideally not happen with the new polarity
};

function App() {
  const [feedbackData, setFeedbackData] = useState([]);
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [sentimentCounts, setSentimentCounts] = useState({
    positive: 0,
    neutral: 0,
    negative: 0,
  });

  useEffect(() => {
    fetch("http://localhost:5000/api/dashboard-data")
      .then((response) => {
        if (!response.ok) {
          throw new Error(
            `Network response was not ok (status: ${response.status})`
          );
        }
        return response.json();
      })
      .then((data) => {
        if (data.feedbackData && data.summary) {
          setFeedbackData(data.feedbackData);
          setSummary(data.summary);

          // Calculate sentiment counts
          let positive = 0;
          let neutral = 0;
          let negative = 0;
          data.feedbackData.forEach((item) => {
            const score = item["Sentiment Score (1-5)"];
            if (score >= 4) positive++;
            else if (score === 3) neutral++;
            else if (score <= 2) negative++;
          });
          setSentimentCounts({ positive, neutral, negative });
        } else {
          console.warn(
            "Data received from API is missing expected fields:",
            data
          );
          setError("Received incomplete data from the server.");
        }
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching dashboard data:", error);
        setError(error.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-sky-900 flex items-center justify-center text-white text-2xl font-semibold animate-pulse">
        Loading Dashboard... ✨
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-red-100 flex flex-col items-center justify-center text-red-700 p-6 text-center">
        <h2 className="text-3xl font-bold mb-3">Oops! Error Loading Data</h2>
        <p className="text-lg mb-1">{error}</p>
        <p className="text-md">
          Please ensure the backend server is running and accessible at
          http://localhost:5000.
        </p>
      </div>
    );
  }

  const barChartData = {
    labels: ["Positive", "Neutral", "Negative"],
    datasets: [
      {
        label: "Feedback Count",
        data: [
          sentimentCounts.positive,
          sentimentCounts.neutral,
          sentimentCounts.negative,
        ],
        backgroundColor: [
          "rgba(75, 192, 192, 0.7)", // Teal
          "rgba(255, 206, 86, 0.7)", // Yellow
          "rgba(255, 99, 132, 0.7)", // Red
        ],
        borderColor: [
          "rgba(75, 192, 192, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(255, 99, 132, 1)",
        ],
        borderWidth: 1,
        borderRadius: 5,
        hoverBackgroundColor: [
          "rgba(75, 192, 192, 0.9)",
          "rgba(255, 206, 86, 0.9)",
          "rgba(255, 99, 132, 0.9)",
        ],
      },
    ],
  };

  const pieChartData = {
    labels: ["Positive", "Neutral", "Negative"],
    datasets: [
      {
        label: "Feedback Distribution",
        data: [
          sentimentCounts.positive,
          sentimentCounts.neutral,
          sentimentCounts.negative,
        ],
        backgroundColor: [
          "rgba(75, 192, 192, 0.8)", // Teal
          "rgba(255, 206, 86, 0.8)", // Yellow
          "rgba(255, 99, 132, 0.8)", // Red
        ],
        borderColor: [
          // Add borders for better separation
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(255, 99, 132, 1)",
        ],
        borderWidth: 2,
        hoverOffset: 8,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#e2e8f0", // slate-200
          font: {
            size: 14,
          },
        },
      },
      title: {
        display: true,
        text: "Feedback Sentiment Analysis",
        color: "#cbd5e1", // slate-300
        font: {
          size: 18,
          weight: "bold",
        },
      },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        titleColor: "#fff",
        bodyColor: "#fff",
        borderColor: "#334155", // slate-700
        borderWidth: 1,
      },
    },
    scales: {
      // Only for bar chart, pie chart doesn't use scales
      y: {
        beginAtZero: true,
        ticks: { color: "#94a3b8" }, // slate-400
        grid: { color: "#334155" }, // slate-700
      },
      x: {
        ticks: { color: "#94a3b8" }, // slate-400
        grid: { color: "#334155" }, // slate-700
      },
    },
  };

  // Stat Card Component
  const StatCard = ({ title, count, color, icon }) => (
    <div
      className={`bg-gradient-to-br ${color} p-6 rounded-xl shadow-2xl transform hover:scale-105 transition-transform duration-300 ease-in-out`}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xl font-semibold text-white">{title}</h3>
        <span className="text-3xl text-white opacity-80">{icon}</span>
      </div>
      <p className="text-5xl font-bold text-white">{count}</p>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-sky-900 text-slate-100 p-4 md:p-8 selection:bg-sky-500 selection:text-white">
      <header className="text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-sky-400 via-cyan-300 to-teal-400 pb-2">
          Customer Feedback Insights
        </h1>
        <p className="text-lg text-sky-200 opacity-90">
          Visualizing sentiment and key themes from your users.
        </p>
      </header>

      {/* Section 1: Stat Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8 mb-12">
        <StatCard
          title="Positive Feedback"
          count={sentimentCounts.positive}
          color="from-green-500 to-emerald-600"
          icon="😊"
        />
        <StatCard
          title="Neutral Feedback"
          count={sentimentCounts.neutral}
          color="from-yellow-500 to-amber-600"
          icon="😐"
        />
        <StatCard
          title="Negative Feedback"
          count={sentimentCounts.negative}
          color="from-red-500 to-rose-600"
          icon="😞"
        />
      </section>

      {/* Section 2: Charts */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8 mb-12">
        <div className="bg-slate-800/70 backdrop-blur-md p-4 md:p-6 rounded-xl shadow-2xl">
          <h2 className="text-2xl font-semibold text-sky-300 mb-4 text-center">
            Feedback Distribution (Bar)
          </h2>
          <div className="h-80 md:h-96">
            {" "}
            {/* Set a fixed height for chart container */}
            <Bar data={barChartData} options={chartOptions} />
          </div>
        </div>
        <div className="bg-slate-800/70 backdrop-blur-md p-4 md:p-6 rounded-xl shadow-2xl">
          <h2 className="text-2xl font-semibold text-sky-300 mb-4 text-center">
            Feedback Distribution (Pie)
          </h2>
          <div className="h-80 md:h-96">
            {" "}
            {/* Set a fixed height for chart container */}
            <Pie
              data={pieChartData}
              options={{ ...chartOptions, scales: {} }}
            />{" "}
            {/* Pie chart doesn't need scales */}
          </div>
        </div>
      </section>

      {/* Section 3: AI Summary */}
      <section className="mb-10 bg-slate-800/70 backdrop-blur-md p-6 rounded-xl shadow-2xl">
        <h2 className="text-2xl md:text-3xl font-semibold text-sky-300 mb-4">
          AI Generated Summary
        </h2>
        <div className="prose prose-invert max-w-none text-slate-300 leading-relaxed">
          {summary.split("\n").map((line, index) => (
            <p key={index} className="mb-2">
              {line}
            </p>
          ))}
        </div>
      </section>

      {/* Optional: Detailed Feedback Table (can be toggled or kept for reference) */}
      <section className="bg-slate-800/70 backdrop-blur-md p-6 rounded-lg shadow-xl">
        <h2 className="text-2xl md:text-3xl font-semibold text-sky-300 mb-6">
          Raw Feedback Data
        </h2>
        {feedbackData.length > 0 ? (
          <div className="overflow-x-auto max-h-96">
            {" "}
            {/* Added max-h for scrollability */}
            <table className="min-w-full divide-y divide-slate-700">
              <thead className="bg-slate-700 sticky top-0">
                {" "}
                {/* Sticky header */}
                <tr>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-sky-300 uppercase tracking-wider"
                  >
                    ID
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-sky-300 uppercase tracking-wider"
                  >
                    Date
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-sky-300 uppercase tracking-wider"
                  >
                    Feedback
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-center text-xs font-medium text-sky-300 uppercase tracking-wider"
                  >
                    Rating
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-center text-xs font-medium text-sky-300 uppercase tracking-wider"
                  >
                    Sentiment
                  </th>
                </tr>
              </thead>
              <tbody className="bg-slate-800 divide-y divide-slate-700">
                {feedbackData.map((item, index) => (
                  <tr
                    key={item["Customer ID"] || index}
                    className="hover:bg-slate-700/50 transition-colors"
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                      {item["Customer ID"]}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                      {item.Date}
                    </td>
                    <td className="px-6 py-4 text-sm text-slate-300 max-w-xs md:max-w-md break-words">
                      {item["Feedback Text"]}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300 text-center">
                      {item.Rating}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300 text-center">
                      {item["Sentiment Score (1-5)"]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-slate-400">
            No feedback data available to display.
          </p>
        )}
      </section>
    </div>
  );
}

export default App;
