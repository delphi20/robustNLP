"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Home, BarChart2, Zap } from "lucide-react"
import ApiStatus from "@/components/api-status"

export default function Navbar() {
  const pathname = usePathname()

  const isActive = (path: string) => {
    return pathname === path ? "bg-blue-700" : ""
  }

  return (
    <nav className="bg-blue-600 text-white shadow-md">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-xl font-bold flex items-center">
            <Zap className="mr-2" />
            robustNLP
          </Link>

          <div className="flex items-center space-x-4">
            <ApiStatus />

            <div className="flex space-x-1">
              <Link href="/" className={`px-3 py-2 rounded-md text-sm font-medium flex items-center ${isActive("/")}`}>
                <Home className="mr-1 h-4 w-4" />
                Home
              </Link>

              <Link
                href="/dashboard"
                className={`px-3 py-2 rounded-md text-sm font-medium flex items-center ${isActive("/dashboard")}`}
              >
                <BarChart2 className="mr-1 h-4 w-4" />
                Dashboard
              </Link>

              <Link
                href="/interactive"
                className={`px-3 py-2 rounded-md text-sm font-medium flex items-center ${isActive("/interactive")}`}
              >
                <Zap className="mr-1 h-4 w-4" />
                Interactive
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
