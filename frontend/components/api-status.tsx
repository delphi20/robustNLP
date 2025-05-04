"use client"

import { useState, useEffect } from "react"
import { isApiAvailable } from "@/utils/constants"
import { AlertCircle, CheckCircle, Loader2 } from "lucide-react"

export default function ApiStatus() {
  const [isAvailable, setIsAvailable] = useState<boolean | null>(null)
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    const checkApi = async () => {
      setChecking(true)
      const available = await isApiAvailable()
      setIsAvailable(available)
      setChecking(false)
    }

    checkApi()

    // Check API status every 30 seconds
    const interval = setInterval(checkApi, 30000)

    return () => clearInterval(interval)
  }, [])

  if (checking) {
    return (
      <div className="flex items-center text-gray-500 text-sm">
        <Loader2 className="h-4 w-4 mr-1 animate-spin" />
        Checking API...
      </div>
    )
  }

  if (isAvailable) {
    return (
      <div className="flex items-center text-green-600 text-sm">
        <CheckCircle className="h-4 w-4 mr-1" />
        API Connected
      </div>
    )
  }

  return (
    <div className="flex items-center text-red-600 text-sm">
      <AlertCircle className="h-4 w-4 mr-1" />
      API Unavailable
    </div>
  )
}
