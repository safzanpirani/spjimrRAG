let activeLangfuseCallbacks: any[] | undefined;
let activeLangfuseTrace: any | undefined;

export const setActiveLangfuseCallbacks = (callbacks?: any[]) => {
  activeLangfuseCallbacks = callbacks;
};

export const getActiveLangfuseCallbacks = (): any[] | undefined => {
  return activeLangfuseCallbacks;
};

export const setActiveLangfuseTrace = (trace?: any) => {
  activeLangfuseTrace = trace;
};

export const getActiveLangfuseTrace = (): any | undefined => {
  return activeLangfuseTrace;
};


