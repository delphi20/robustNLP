// API URL for the backend service
export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

// Helper function to check if the API is available
export async function isApiAvailable(): Promise<boolean> {
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout

    const response = await fetch(`${API_URL}/`, {
      method: "GET",
      signal: controller.signal,
    })

    clearTimeout(timeoutId)
    return response.ok
  } catch (error) {
    console.warn("API availability check failed:", error)
    return false
  }
}
