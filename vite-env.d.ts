/// <reference types="vite/client" />

// Raw-text imports (Vite `?raw`). Used to load js-aruco2's cv.js source and
// evaluate it against a controlled `this` context — see colorcalib.ts loadCV().
declare module '*?raw' {
  const src: string;
  export default src;
}
