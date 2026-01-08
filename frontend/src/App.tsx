import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const sampleSeries = [
  { name: "00:00", value: 120 },
  { name: "06:00", value: 240 },
  { name: "12:00", value: 320 },
  { name: "18:00", value: 200 }
];

export default function App() {
  return (
    <div className="min-h-screen px-6 py-8">
      <header className="mb-8">
        <p className="text-sm uppercase tracking-[0.2em] text-slate-400">Operations</p>
        <h1 className="text-3xl font-semibold text-white">Realtime incident overview</h1>
        <p className="mt-2 max-w-2xl text-slate-300">
          Vite + React + TypeScript scaffold with Tailwind utilities, Recharts visualization,
          and Leaflet map tooling.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <section className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6">
          <h2 className="mb-4 text-lg font-medium text-white">Call volume</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sampleSeries}>
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#38bdf8" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6">
          <h2 className="mb-4 text-lg font-medium text-white">Incident map</h2>
          <div className="h-64 overflow-hidden rounded-xl">
            <MapContainer center={[39.9526, -75.1652]} zoom={11} className="h-full w-full">
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <Marker position={[39.9526, -75.1652]}>
                <Popup>Sample incident marker</Popup>
              </Marker>
            </MapContainer>
          </div>
        </section>
      </div>
    </div>
  );
}
