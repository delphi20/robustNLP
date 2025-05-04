// Add a health check endpoint for local development
import { NextResponse } from "next/server"

export async function GET() {
  return NextResponse.json({ status: "ok" })
}
