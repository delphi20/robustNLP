"use client"

import { useState, useEffect } from "react"
import Head from "next/head"
import { API_URL, isApiAvailable } from "@/utils/constants"
import Loading from "@/components/loading"
import { AlertCircle } from "lucide-react"  


export default function Dashboard() {
  const [metrics, setMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [plots, setPlots] = useState<Record<string, string>>({})
  const [apiAvailable, setApiAvailable] = useState<boolean | null>(null)

  useEffect(() => {
    // First check if API is available
    const checkApiAndFetchData = async () => {
      const available = await isApiAvailable()
      setApiAvailable(available)

      if (!available) {
        setError("API server is not available. Please ensure the backend is running.")
        setLoading(false)
        return
      }

      fetchEvaluationData()
    }

    checkApiAndFetchData()
  }, [])

  const fetchEvaluationData = async () => {
    try {
      setLoading(true)

      const response = await fetch(`${API_URL}/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dataset_name: "imdb",
          num_samples: 50,
        }),
        // Add timeout
        signal: AbortSignal.timeout(30000), // 30 second timeout
      })

      if (!response.ok) {
        throw new Error(`Failed to fetch evaluation data: API returned status ${response.status}`)
      }

      const data = await response.json()
      setMetrics(data.metrics)

      // Generate plots
        const plotTypes = ["attack_success_rate", "defense_efficacy", "accuracy_comparison"]
        const plotPromises = plotTypes.map(async (plotType) => {
          let plotData = {}

          if (plotType === "attack_success_rate") {
            plotData = {
              labels: Object.keys(data.metrics),
              values: Object.keys(data.metrics).map((key) => data.metrics[key].attack_success_rate),
              title: "Attack Success Rate by Method",
              xlabel: "Attack Method",
              ylabel: "Success Rate",
            }
          } else if (plotType === "defense_efficacy") {
            plotData = {
              labels: Object.keys(data.metrics),
              values: Object.keys(data.metrics).map((key) => data.metrics[key].defense_efficacy),
              title: "Defense Efficacy by Attack Method",
              xlabel: "Attack Method",
              ylabel: "Defense Efficacy",
            }
          } else if (plotType === "accuracy_comparison") {
            plotData = {
              data: [
                ...Object.keys(data.metrics).map((key) => ({
                  attackMethod: key,
                  model: "Original",
                  accuracy: data.metrics[key].original_accuracy,
                })),
                ...Object.keys(data.metrics).map((key) => ({
                  attackMethod: key,
                  model: "Defended",
                  accuracy: data.metrics[key].defended_accuracy,
                })),
              ],
              x: "attackMethod",
              y: "accuracy",
              hue: "model",
              title: "Accuracy Comparison: Original vs. Defended Model",
              xlabel: "Attack Method",
              ylabel: "Accuracy",
            }
          }

        const plotResponse = await fetch(`${API_URL}/plot`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            plot_type: plotType === "accuracy_comparison" ? "comparison" : "bar",
            data: plotData,
          }),
          // Add timeout
          signal: AbortSignal.timeout(20000), // 20 second timeout
        })

        if (!plotResponse.ok) {
          throw new Error(`Failed to generate ${plotType} plot: API returned status ${plotResponse.status}`)
        }

        const plotResult = await plotResponse.json()
        return { [plotType]: plotResult.plot_data }
      })

      const plotResults = await Promise.all(plotPromises)
      const allPlots = plotResults.reduce((acc, plot) => ({ ...acc, ...plot }), {})
      setPlots(allPlots)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <Loading message="Loading evaluation data..." />
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 mr-2" />
            <p>Error: {error}</p>
          </div>
          <p className="mt-2">Please ensure the backend API is running.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>Dashboard | NLP Defense System</title>
      </Head>

      <main>
        <div className="bg-blue-600 text-white shadow-md">
          <div className="container mx-auto px-4 py-4">
            <h1 className="text-2xl font-bold">Evaluation Dashboard</h1>
          </div>
        </div>

        <div className="container mx-auto px-4 py-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {Object.keys(metrics).map((attackMethod) => (
              <div key={attackMethod} className="bg-white rounded-lg shadow-md p-4">
                <h2 className="text-lg font-semibold mb-2">
                  Attack Method: <span className="text-blue-600">{attackMethod}</span>
                </h2>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-gray-50 p-2 rounded">
                    <p className="text-sm text-gray-500">Attack Success Rate</p>
                    <p className="text-lg font-medium">
                      {(metrics[attackMethod].attack_success_rate * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-gray-50 p-2 rounded">
                    <p className="text-sm text-gray-500">Defense Efficacy</p>
                    <p className="text-lg font-medium">{(metrics[attackMethod].defense_efficacy * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-gray-50 p-2 rounded">
                    <p className="text-sm text-gray-500">Original Accuracy</p>
                    <p className="text-lg font-medium">{(metrics[attackMethod].original_accuracy * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-gray-50 p-2 rounded">
                    <p className="text-sm text-gray-500">Defended Accuracy</p>
                    <p className="text-lg font-medium">{(metrics[attackMethod].defended_accuracy * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.keys(plots).map((plotKey) => (
              <div key={plotKey} className="bg-white rounded-lg shadow-md p-4">
                <h2 className="text-lg font-semibold mb-4 text-center">
                  {plotKey
                    .split("_")
                    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(" ")}
                </h2>
                <div className="flex justify-center">
                  <img
                    src={`data:image/png;base64,${plots[plotKey]}`}
                    alt={`${plotKey} visualization`}
                    className="max-w-full h-auto"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}
