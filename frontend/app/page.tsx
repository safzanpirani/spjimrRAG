import { ChatInterface } from "@/components/chat-interface"
import { Header } from "@/components/header"
import { Footer } from "@/components/footer"

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="flex-1 flex flex-col">
        <ChatInterface />
      </main>
      <Footer />
    </div>
  )
}
