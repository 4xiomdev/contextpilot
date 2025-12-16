"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Label } from "@/components/ui/label";

export default function SignInPage() {
  return (
    <Card className="border-zinc-800 bg-zinc-900/50">
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl font-bold">Sign in</CardTitle>
        <CardDescription>
          Enter your email below to access your account
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input id="email" placeholder="m@example.com" type="email" className="bg-zinc-800/50 border-zinc-700" />
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="password">Password</Label>
            <Link href="#" className="text-sm text-violet-400 hover:text-violet-300">
              Forgot password?
            </Link>
          </div>
          <Input id="password" type="password" className="bg-zinc-800/50 border-zinc-700" />
        </div>
        <Button className="w-full bg-violet-600 hover:bg-violet-700">
          Sign In
        </Button>
      </CardContent>
      <CardFooter className="flex flex-col gap-2">
        <div className="text-sm text-zinc-500 text-center">
          Don&apos;t have an account?{" "}
          <Link href="/sign-up" className="text-violet-400 hover:text-violet-300">
            Sign up
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
