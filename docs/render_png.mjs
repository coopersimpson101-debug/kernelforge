// Render docs/assets/*.svg -> .png with resvg (npm i @resvg/resvg-js):  node docs/render_png.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { Resvg } from '@resvg/resvg-js';

for (const [name, width] of [['banner', 1280], ['bench', 1000]]) {
  const svg = readFileSync(new URL(`./assets/${name}.svg`, import.meta.url), 'utf8');
  const png = new Resvg(svg, { fitTo: { mode: 'width', value: width }, font: { loadSystemFonts: true } }).render().asPng();
  writeFileSync(new URL(`./assets/${name}.png`, import.meta.url), png);
  console.log(`docs/assets/${name}.png ${png.length} bytes`);
}
