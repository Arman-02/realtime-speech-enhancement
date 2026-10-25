/** Perceptually uniform "magma" colormap (polynomial fit), as a 256-entry RGB lookup table. */
const C: [number, number, number][] = [
  [-0.002136485053939582, -0.000749655052795221, -0.005386127855323933],
  [0.2516605407371642, 0.6775232436837668, 2.494026599312351],
  [8.353717279216625, -3.577719514958484, 0.3144679030132573],
  [-27.66873308576866, 14.26473078096533, -13.64921318813922],
  [52.17613981234068, -27.94360607168351, 12.94416944238394],
  [-50.76852536473588, 29.04658282127291, 4.23415299384598],
  [18.65570506591883, -11.48977351997711, -5.601961508734096],
];

function evalPoly(t: number, ch: 0 | 1 | 2): number {
  let v = C[6][ch];
  for (let i = 5; i >= 0; i--) v = C[i][ch] + t * v;
  return Math.max(0, Math.min(1, v));
}

export const MAGMA: Uint8ClampedArray = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    lut[i * 3] = Math.round(evalPoly(t, 0) * 255);
    lut[i * 3 + 1] = Math.round(evalPoly(t, 1) * 255);
    lut[i * 3 + 2] = Math.round(evalPoly(t, 2) * 255);
  }
  return lut;
})();

export function magmaCss(t: number): string {
  const i = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
  return `rgb(${MAGMA[i]} ${MAGMA[i + 1]} ${MAGMA[i + 2]})`;
}
