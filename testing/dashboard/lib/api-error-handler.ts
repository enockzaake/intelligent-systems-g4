// API Error Handler
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: any
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export async function handleApiResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `API Error: ${response.status}`
    let errorDetails = null

    try {
      const errorData = await response.json()
      errorMessage = errorData.detail || errorData.message || errorMessage
      errorDetails = errorData
    } catch {
      errorMessage = `HTTP ${response.status}: ${response.statusText}`
    }

    throw new ApiError(errorMessage, response.status, errorDetails)
  }

  try {
    return await response.json()
  } catch (error) {
    throw new ApiError('Failed to parse response JSON', response.status)
  }
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unknown error occurred'
}

