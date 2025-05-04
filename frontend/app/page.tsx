import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Shield, BarChart3, Zap, ArrowRight, FileText, Code } from "lucide-react"

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      {/* <header className="sticky top-0 z-10 border-b bg-primary text-primary-foreground">
        <div className="flex h-20 items-center px-4 md:px-6 lg:px-8">
          <div className="flex items-center gap-2">
            <Shield className="h-6 w-6" />
            <h1 className="text-xl font-bold">robustNLP</h1>
          </div>
          <nav className="ml-auto hidden gap-6 md:flex">
            <Link href="/" className="text-sm font-medium underline underline-offset-4">
              Home
            </Link>
            <Link href="/dashboard" className="text-sm font-medium hover:underline hover:underline-offset-4">
              Dashboard
            </Link>
            <Link href="/interactive" className="text-sm font-medium hover:underline hover:underline-offset-4">
              Interactive
            </Link>
            <Link href="/documentation" className="text-sm font-medium hover:underline hover:underline-offset-4">
              Documentation
            </Link>
          </nav>
        </div>
      </header> */}

      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 bg-muted/40">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <div className="inline-flex items-center rounded-lg bg-primary px-3 py-1 text-sm text-primary-foreground">
                  <Shield className="mr-1 h-4 w-4" />
                  <span>Robust NLP Defense System</span>
                </div>
                <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
                  Defending NLP Models Against Adversarial Attacks
                </h1>
                <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
                  A comprehensive platform for evaluating and improving the robustness of natural language processing
                  models against various adversarial text attacks.
                </p>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <Link href="/dashboard">
                  <Button size="lg">
                    View Dashboard
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/interactive">
                  <Button variant="outline" size="lg">
                    Try Interactive Testing
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className="w-full py-12 md:py-24 lg:py-32">
          <div className="container px-4 md:px-6">
            <div className="mx-auto grid max-w-5xl items-center gap-6 lg:grid-cols-2 lg:gap-12">
              <div className="space-y-4">
                <div className="inline-flex items-center rounded-lg bg-muted px-3 py-1 text-sm">
                  <BarChart3 className="mr-1 h-4 w-4" />
                  <span>Comprehensive Evaluation</span>
                </div>
                <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">
                  Evaluate Model Robustness Against Multiple Attack Types
                </h2>
                <p className="text-muted-foreground md:text-lg">
                  Our system evaluates NLP models against textual adversarial attacks, word substitution methods, and
                  character manipulation techniques to provide a comprehensive assessment of model robustness.
                </p>
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                <Card>
                  <CardHeader className="pb-2">
                    <Shield className="h-5 w-5 text-primary" />
                    <CardTitle className="mt-2">Defense Mechanisms</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Implement and evaluate various defense strategies to protect your models from adversarial attacks.
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    <CardTitle className="mt-2">Performance Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Track key metrics like attack success rate, defense efficacy, and model accuracy before and after
                      defense.
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <Zap className="h-5 w-5 text-primary" />
                    <CardTitle className="mt-2">Interactive Testing</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Test your own text inputs against various attack methods and see how defenses perform in
                      real-time.
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <FileText className="h-5 w-5 text-primary" />
                    <CardTitle className="mt-2">Detailed Reports</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Generate comprehensive reports with visualizations to understand model vulnerabilities and
                      improvements.
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </section>

        <section className="w-full py-12 md:py-24 lg:py-32 bg-muted/40">
          <div className="container px-4 md:px-6">
            <div className="mx-auto max-w-5xl space-y-6">
              <div className="space-y-2 text-center">
                <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">Get Started</h2>
                <p className="text-muted-foreground md:text-lg">
                  Choose a module to begin working with the NLP Defense System
                </p>
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                <Link href="/dashboard" className="block">
                  <Card className="h-full transition-all hover:shadow-lg">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-primary" />
                        Evaluation Dashboard
                      </CardTitle>
                      <CardDescription>Comprehensive metrics and visualizations for model evaluation</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-muted-foreground">
                        View attack success rates, defense efficacy metrics, and performance visualizations across
                        multiple attack methods.
                      </p>
                    </CardContent>
                    <CardFooter>
                      <Button variant="ghost" className="w-full justify-between">
                        Open Dashboard
                        <ArrowRight className="h-4 w-4" />
                      </Button>
                    </CardFooter>
                  </Card>
                </Link>
                <Link href="/interactive" className="block">
                  <Card className="h-full transition-all hover:shadow-lg">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Code className="h-5 w-5 text-primary" />
                        Interactive Testing
                      </CardTitle>
                      <CardDescription>Test attacks and defenses on custom inputs</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-muted-foreground">
                        Test adversarial attacks and defenses on your own text inputs to see how they perform in
                        real-world scenarios.
                      </p>
                    </CardContent>
                    <CardFooter>
                      <Button variant="ghost" className="w-full justify-between">
                        Try Interactive Testing
                        <ArrowRight className="h-4 w-4" />
                      </Button>
                    </CardFooter>
                  </Card>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t bg-background">
        <div className="container flex flex-col items-center justify-between gap-4 py-10 md:h-24 md:flex-row md:py-0">
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            <p className="text-sm text-muted-foreground">robustNLP Defense System</p>
          </div>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <Link href="/documentation" className="hover:underline">
              Documentation
            </Link>
            <Link href="/about" className="hover:underline">
              About
            </Link>
            <Link href="/contact" className="hover:underline">
              Contact
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
