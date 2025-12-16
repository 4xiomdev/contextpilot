"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Label } from "@/components/ui/label";

export default function SignUpPage() {
  return (
    <Card className="border-zinc-800 bg-zinc-900/50">
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl font-bold">Create an account</CardTitle>
        <CardDescription>
          Enter your details to get started with ContextPilot
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="first-name">First name</Label>
            <Input id="first-name" placeholder="John" className="bg-zinc-800/50 border-zinc-700" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="last-name">Last name</Label>
            <Input id="last-name" placeholder="Doe" className="bg-zinc-800/50 border-zinc-700" />
          </div>
        </div>
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input id="email" placeholder="m@example.com" type="email" className="bg-zinc-800/50 border-zinc-700" />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <Input id="password" type="password" className="bg-zinc-800/50 border-zinc-700" />
        </div>
        <Button className="w-full bg-violet-600 hover:bg-violet-700">
          Create account
        </Button>
      </CardContent>
      <CardFooter className="flex flex-col gap-2">
        <div className="text-sm text-zinc-500 text-center">
          Already have an account?{" "}
          <Link href="/sign-in" className="text-violet-400 hover:text-violet-300">
            Sign in
          </Link>
        </div>
        <div className="text-xs text-zinc-600 text-center mt-4">
          <Link href="/" className="hover:text-zinc-400">
            ← Back to Home
          </Link>
        </div>
      </CardFooter>
    </Card>
  );
}
