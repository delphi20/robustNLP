"use client"

import { useState, useEffect } from "react"
import Head from "next/head"
import { API_URL, isApiAvailable } from "@/utils/constants"
import { AlertCircle } from "lucide-react"

export default function Interactive() {
  const [inputText, setInputText] = useState("")
  const [label, setLabel] = useState("1") // Positive sentiment by default
  const [attackMethod, setAttackMethod] = useState("textfooler")
  const [usePreprocessing, setUsePreprocessing] = useState(true)
  const [loading, setLoading] = useState(false)
  const [attackResults, setAttackResults] = useState<any>(null)
  const [defenseResults, setDefenseResults] = useState<any>(null)
  const [availableAttacks, setAvailableAttacks] = useState(["textfooler", "bert-attack"])
  const [apiAvailable, setApiAvailable] = useState<boolean | null>(null)

  // Check API availability on component mount
  useEffect(() => {
    const checkApi = async () => {
      const available = await isApiAvailable()
      setApiAvailable(available)
    }
    checkApi()
  }, [])

  // Fetch available attack methods on component mount
  useEffect(() => {
    const fetchAttackMethods = async () => {
      try {
        // First check if the API is available
        if (!apiAvailable) {
          console.warn("API is not available. Using default attack methods.")
          // Use default values if API is not available
          setAvailableAttacks(["textfooler", "bert-attack", "deepwordbug"])
          setAttackMethod("textfooler")
          return
        }

        const response = await fetch(`${API_URL}/attacks`, {
          // Add timeout to prevent hanging requests
          signal: AbortSignal.timeout(10000), // 10 second timeout
        })

        if (!response.ok) {
          throw new Error(`API returned status ${response.status}`)
        }

        const data = await response.json()
        if (data.attack_methods && Array.isArray(data.attack_methods)) {
          setAvailableAttacks(data.attack_methods)
          if (data.attack_methods.length > 0) {
            setAttackMethod(data.attack_methods[0])
          }
        } else {
          throw new Error("Invalid response format")
        }
      } catch (error) {
        console.error("Failed to fetch attack methods:", error)
        // Use default values if fetch fails
        setAvailableAttacks(["textfooler", "bert-attack", "deepwordbug"])
        setAttackMethod("textfooler")
      }
    }

    if (apiAvailable !== null) {
      fetchAttackMethods()
    }
  }, [apiAvailable])

  // Generate attack function
  const generateAttack = async () => {
    if (!inputText.trim()) {
      alert("Please enter some text to attack")
      return
    }

    setLoading(true)
    try {
      // Check if API is available
      if (!apiAvailable) {
        throw new Error("API server is not available. Please ensure the backend is running.")
      }

      const response = await fetch(`${API_URL}/generate-attack`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          texts: [inputText],
          labels: [Number.parseInt(label)],
          attack_method: attackMethod,
        }),
        // Add timeout to prevent hanging requests
        signal: AbortSignal.timeout(30000), // 30 second timeout for potentially long operation
      })

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`)
      }

      const data = await response.json()
      setAttackResults(data)

      // If attack was generated, evaluate defense
      if (data.adversarial_text && data.adversarial_text.length > 0) {
        await evaluateDefense(data.adversarial_text[0])
      }
    } catch (error: any) {
      console.error("Error generating attack:", error)
      alert(`Error: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Evaluate defense function
  const evaluateDefense = async (textToDefend?: string) => {
    try {
      // Check if API is available
      if (!apiAvailable) {
        throw new Error("API server is not available. Please ensure the backend is running.")
      }

      const response = await fetch(`${API_URL}/defend`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          texts: [textToDefend || inputText],
          use_preprocessing: usePreprocessing,
        }),
        // Add timeout
        signal: AbortSignal.timeout(20000), // 20 second timeout
      })

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`)
      }

      const data = await response.json()
      setDefenseResults(data)
    } catch (error: any) {
      console.error("Error evaluating defense:", error)
      alert(`Error: ${error.message}`)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>Interactive Testing | NLP Defense System</title>
      </Head>

      <main>
        <div className="bg-blue-600 text-white shadow-md">
          <div className="container mx-auto px-4 py-4">
            <h1 className="text-2xl font-bold">Interactive Testing</h1>
          </div>
        </div>

        <div className="container mx-auto px-4 py-6">
          {/* API Status Warning */}
          {apiAvailable === false && (
            <div className="mb-6 bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 mr-2" />
                <p>
                  <span className="font-bold">Warning:</span> API server is not available. Please ensure the backend is
                  running.
                </p>
              </div>
            </div>
          )}

          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Generate an Adversarial Attack</h2>

            <div className="mb-4">
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="inputText">
                Input Text
              </label>
              <textarea
                id="inputText"
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                rows={4}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text to attack (e.g., 'This movie was fantastic, I really enjoyed it.')"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="label">
                  Sentiment Label
                </label>
                <select
                  id="label"
                  className="shadow border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                >
                  <option value="1">Positive</option>
                  <option value="0">Negative</option>
                </select>
              </div>

              <div>
                <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="attackMethod">
                  Attack Method
                </label>
                <select
                  id="attackMethod"
                  className="shadow border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                  value={attackMethod}
                  onChange={(e) => setAttackMethod(e.target.value)}
                >
                  {availableAttacks.map((method) => (
                    <option key={method} value={method}>
                      {method}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="preprocessing">
                  Use Preprocessing Defense
                </label>
                <div className="mt-2">
                  <label className="inline-flex items-center">
                    <input
                      type="checkbox"
                      className="form-checkbox h-5 w-5 text-blue-600"
                      checked={usePreprocessing}
                      onChange={(e) => setUsePreprocessing(e.target.checked)}
                    />
                    <span className="ml-2 text-gray-700">Enable preprocessing</span>
                  </label>
                </div>
              </div>
            </div>

            <div className="flex justify-center">
              <button
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                type="button"
                onClick={generateAttack}
                disabled={loading || apiAvailable === false}
              >
                {loading ? "Processing..." : "Generate Adversarial Example"}
              </button>
            </div>
          </div>

          {attackResults && (
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-xl font-semibold mb-4">Attack Results</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium mb-2">Original Text</h3>
                  <div className="border rounded-lg p-4 bg-gray-50">
                    <p>{attackResults.original_text[0]}</p>
                  </div>
                  <p className="mt-2">
                    <span className="font-medium">Original Label:</span>{" "}
                    {attackResults.original_label[0] === 1 ? "Positive" : "Negative"}
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-2">Adversarial Text</h3>
                  <div className="border rounded-lg p-4 bg-gray-50">
                    <p>{attackResults.adversarial_text[0]}</p>
                  </div>
                  <p className="mt-2">
                    <span className="font-medium">Attack Success:</span>{" "}
                    {attackResults.success[0] ? "Successful" : "Failed"}
                  </p>
                </div>
              </div>
            </div>
          )}

          {defenseResults && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Defense Results</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium mb-2">Original Model</h3>
                  <div className="bg-gray-50 border rounded-lg p-4">
                    <p>
                      <span className="font-medium">Prediction:</span>{" "}
                      {defenseResults.original_predictions[0] === 1 ? "Positive" : "Negative"}
                    </p>
                    <div className="mt-4">
                      <p className="font-medium mb-1">Confidence Scores:</p>
                      <div className="flex h-6 overflow-hidden rounded-lg bg-gray-200">
                        <div
                          className="bg-red-500 flex justify-center items-center text-xs text-white"
                          style={{ width: `${defenseResults.original_scores[0][0] * 100}%` }}
                        >
                          {(defenseResults.original_scores[0][0] * 100).toFixed(1)}%
                        </div>
                        <div
                          className="bg-green-500 flex justify-center items-center text-xs text-white"
                          style={{ width: `${defenseResults.original_scores[0][1] * 100}%` }}
                        >
                          {(defenseResults.original_scores[0][1] * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="flex justify-between text-xs mt-1">
                        <span>Negative</span>
                        <span>Positive</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-2">Defended Model</h3>
                  <div className="bg-gray-50 border rounded-lg p-4">
                    <p>
                      <span className="font-medium">Prediction:</span>{" "}
                      {defenseResults.defended_predictions[0] === 1 ? "Positive" : "Negative"}
                    </p>
                    <div className="mt-4">
                      <p className="font-medium mb-1">Confidence Scores:</p>
                      <div className="flex h-6 overflow-hidden rounded-lg bg-gray-200">
                        <div
                          className="bg-red-500 flex justify-center items-center text-xs text-white"
                          style={{ width: `${defenseResults.defended_scores[0][0] * 100}%` }}
                        >
                          {(defenseResults.defended_scores[0][0] * 100).toFixed(1)}%
                        </div>
                        <div
                          className="bg-green-500 flex justify-center items-center text-xs text-white"
                          style={{ width: `${defenseResults.defended_scores[0][1] * 100}%` }}
                        >
                          {(defenseResults.defended_scores[0][1] * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="flex justify-between text-xs mt-1">
                        <span>Negative</span>
                        <span>Positive</span>
                      </div>
                    </div>

                    {defenseResults.preprocessed_texts && (
                      <div className="mt-4">
                        <p className="font-medium">Preprocessed Text:</p>
                        <p className="text-sm mt-1 italic">{defenseResults.preprocessed_texts[0]}</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
