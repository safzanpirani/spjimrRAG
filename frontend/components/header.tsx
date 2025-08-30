import Image from "next/image"

export function Header() {
  return (
    <header className="border-b border-border bg-card">
      <div className="max-w-4xl mx-auto px-4 py-6">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <Image src="/spjimr-logo-purple.png" alt="SPJIMR Logo" width={120} height={40} className="h-10 w-auto" />
            <div className="w-px h-10 bg-border" />
            <Image src="/pgpm-logo.svg" alt="PGPM Logo" width={100} height={30} className="hidden sm:block h-8 w-auto" />
          </div>
          <div className="ml-auto self-end mb-1">
            <div className="text-right space-y-0">
              <p className="text-xs text-transparent">.</p>
              <h1 className="text-lg font-semibold text-primary leading-none">PGPM Assistant</h1>
              <p className="text-xs text-muted-foreground">Ask about admissions, curriculum & more</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
