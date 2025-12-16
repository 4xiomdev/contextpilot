import { Navigation } from "@/components/navigation";

export default function AppLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="flex min-h-screen">
      <Navigation />
      <main className="flex-1 ml-64 p-8">{children}</main>
    </div>
  );
}
