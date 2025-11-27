'use client';

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { AlertCircle, Zap, Timer, TrendingUp, Brain, MessageSquare } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import Link from "next/link";
import { ArrowUpRight, Trash2 } from "lucide-react";
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

interface Metrics {
  resolveRate: number;
  escalateRate: number;
  totalTickets: number;
  resolved: number;
  escalated: number;
  confidenceTrend: { date: string; rate: number }[]; 
  tagBreakdown: { name: string; value: number; color: string }[]; 
}

interface Ticket {
  id: number;
  summary: string;
  priority: string;
  department: string;
  tag: string;
}

interface KBEntry {
  id: number;
  title: string;
  content: string;
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [escalatedTickets, setEscalatedTickets] = useState<Ticket[]>([]);
  const [kbEntries, setKBEntries] = useState<KBEntry[]>([]);

  

  const fetchMetrics = async () => {
    try {
      setError(null);
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const res = await fetch(`${backendUrl}/metrics`);
      if (!res.ok) throw new Error("Failed to fetch metrics");
      const data = await res.json();

      setMetrics({
        resolveRate: data.resolveRate || 0,
        escalateRate: data.escalateRate || 0,
        totalTickets: data.totalTickets || 0,
        resolved: data.resolved || 0,
        escalated: data.escalated || 0,
        confidenceTrend: data.confidenceTrend || [ 
          { date: "Nov 17", rate: 74 }, { date: "Nov 18", rate: 76 },
          { date: "Nov 19", rate: 78 }, { date: "Nov 20", rate: 80 },
          { date: "Nov 21", rate: 82 }, { date: "Nov 22", rate: data.resolveRate || 0 }
        ],
        tagBreakdown: data.tagBreakdown || [ 
          { name: "Refund", value: 24, color: "#f43f5e" },
          { name: "Bug", value: 16, color: "#8b5cf6" },
          { name: "Shipping", value: 26, color: "#3b82f6" },
          { name: "Billing", value: 14, color: "#10b981" },
          { name: "Other", value: 20, color: "#6b7280" }
        ]
      });
    } catch (e) {
      setError("Failed to fetch metrics—check if backend is running or logs for issues.");
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  };

  const fetchEscalatedTickets = async () => {
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const res = await fetch(`${backendUrl}/escalate`);
      if (!res.ok) throw new Error("Failed to fetch escalated tickets");
      const data = await res.json();
      setEscalatedTickets(data.items || []);  // Real tickets or fallback empty array
    } catch (e) {
      setError("Failed to fetch escalated tickets—check backend.");
      setEscalatedTickets([]);  // Fallback empty on error
    }
  };

  const fetchKBEntries = async () => {
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const res = await fetch(`${backendUrl}/kb`);
      if (!res.ok) throw new Error("Failed to fetch KB entries");
      const data = await res.json();
      setKBEntries(data.items || []);  // Real entries or fallback empty
    } catch (e) {
      setError("Failed to fetch KB entries—check backend.");
      setKBEntries([]);
    }
  };

  const saveKB = async () => {
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const res = await fetch(`${backendUrl}/kb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Title', content: 'New Content' })  // Replace with form input
      });
      if (!res.ok) throw new Error("Failed to save KB");
      fetchKBEntries();  // Refresh list
    } catch (e) {
      setError("Failed to save KB—check backend.");
    }
  };    

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchMetrics();
    fetchKBEntries();
    fetchEscalatedTickets();  // Added
    const interval = setInterval(() => {
      fetchMetrics();
      fetchEscalatedTickets();  // Added
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="text-2xl font-bold flex items-center gap-3">
          <Zap className="w-8 h-8 animate-pulse text-yellow-500" />
          Loading the empire...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <Zap className="w-10 h-10 text-yellow-500" />
            Support Agent Dashboard
            <Badge variant="secondary" className="ml-4 text-lg">
              Auto-Resolve: {(metrics?.resolveRate || 0).toFixed(1)}%
            </Badge>
          </h1>
          <p className="text-muted-foreground mt-2">Built by the support team.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="border-green-500/20">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Auto-Resolve Rate</CardTitle>
              <TrendingUp className="w-5 h-5 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-500">{(metrics?.resolveRate || 0).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">+9.7% this week</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Escalation Rate</CardTitle>
              <TrendingUp className="w-5 h-5 text-red-500" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-500">{(metrics?.escalateRate || 0).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">-4.2% this week</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Total Tickets</CardTitle>
              <MessageSquare className="w-5 h-5 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{metrics?.totalTickets || 0}</div>
              <p className="text-xs text-muted-foreground">{metrics?.escalated || 0} escalated</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Resolved Tickets</CardTitle>
              <MessageSquare className="w-5 h-5 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{metrics?.resolved || 0}</div>
              <p className="text-xs text-muted-foreground">{metrics?.escalated || 0} escalated</p>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="metrics" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:inline-flex">
            <TabsTrigger value="metrics">Live Metrics</TabsTrigger>
            <TabsTrigger value="kb">Knowledge Base</TabsTrigger>
            <TabsTrigger value="queue">Escalation Queue</TabsTrigger>
          </TabsList>

          <TabsContent value="metrics" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader><CardTitle>30-Day Auto-Resolve Trend</CardTitle></CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics?.confidenceTrend}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="rate" stroke="#10b981" strokeWidth={3} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle>Ticket Breakdown</CardTitle></CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie data={metrics?.tagBreakdown} cx="50%" cy="50%" innerRadius={60} outerRadius={100} dataKey="value">
                        {metrics?.tagBreakdown.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="kb">
            <Card>
              <CardHeader><CardTitle>Knowledge Base Editor</CardTitle></CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {kbEntries.map((entry, index) => (
                    <div key={entry.id} className="p-4 border rounded">
                      <h3>{entry.title}</h3>
                      <p>{entry.content}</p>
                    </div>
                  ))}
                </div>
                <Button onClick={saveKB}>Save and Rebuild Vector DB</Button>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="queue">
            <Card>
              <CardHeader><CardTitle>Escalation Queue</CardTitle></CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>ID</TableHead>
                      <TableHead>Summary</TableHead>
                      <TableHead>Priority</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Tag</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {escalatedTickets.map((ticket) => (
                      <TableRow key={ticket.id}>
                        <TableCell>{ticket.id}</TableCell>
                        <TableCell>{ticket.summary}</TableCell>
                        <TableCell>
                          <Badge variant={ticket.priority === 'Urgent' ? 'destructive' : 'default'}>{ticket.priority}</Badge>
                        </TableCell>
                        <TableCell>{ticket.department}</TableCell>
                        <TableCell>{ticket.tag}</TableCell>
                        <TableCell>
                          <Button variant="ghost" size="icon" asChild>
                            <Link href={`/escalate/${ticket.id}`}>
                              <ArrowUpRight className="h-4 w-4" />
                            </Link>
                          </Button>
                          <Button variant="ghost" size="icon">
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

      </div>
    </div>
  );
}