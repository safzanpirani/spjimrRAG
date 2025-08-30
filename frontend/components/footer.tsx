import { ExternalLink } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t border-border bg-muted/30 py-6">
      <div className="max-w-4xl mx-auto px-4">
        <div className="flex justify-center items-center text-sm text-muted-foreground">
          <a
            href="https://www.spjimr.org"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 hover:text-primary transition-colors"
          >
            Visit SPJIMR Website
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </div>
    </footer>
  )
}
