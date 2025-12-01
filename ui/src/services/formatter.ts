/**
 * Make URLs in references clickable without changing the citation format.
 * Only converts URLs to markdown links, preserving the original citation text.
 */
export function convertReferencesToLinks(content: string): string {

    // Skip URLs that are already in markdown link format [text](url)
    const urlPattern = /(https?:\/\/[^\s\)\n]+)/g
    let lastIndex = 0
    let result = ''
    
    let match
    while ((match = urlPattern.exec(content)) !== null) {
      const url = match[0]
      const matchStart = match.index
      const matchEnd = matchStart + url.length
      
      // Add text before the URL
      result += content.substring(lastIndex, matchStart)
      
      // Check if URL is already part of a markdown link
      const before = content.substring(Math.max(0, matchStart - 2), matchStart)
      const after = content.substring(matchEnd, matchEnd + 1)
      
      // If URL is already in a markdown link (preceded by ]( or followed by )), don't convert
      if (before === '](' || after === ')') {
        result += url
      } else {
        result += `[${url}](${url})`
      }
      
      lastIndex = matchEnd
    }
    
    // Add remaining text
    result += content.substring(lastIndex)
    
    return result
  }