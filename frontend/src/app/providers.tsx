import type { PropsWithChildren } from "react";
import { createContext, useContext } from "react";

import { createApiClient, type ApiClient } from "./api/client";

const defaultApiClient = createApiClient({
  baseUrl: import.meta.env.VITE_API_BASE_URL || undefined,
});

const ApiClientContext = createContext<ApiClient>(defaultApiClient);

export interface AppProvidersProps extends PropsWithChildren {
  apiClient?: ApiClient;
}

export function AppProviders({ apiClient, children }: AppProvidersProps) {
  return (
    <ApiClientContext.Provider value={apiClient ?? defaultApiClient}>
      {children}
    </ApiClientContext.Provider>
  );
}

export function useApiClient(): ApiClient {
  return useContext(ApiClientContext);
}
