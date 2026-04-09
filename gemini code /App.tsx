import React, { useState, useEffect, useRef } from 'react';
import { 
  Scale, 
  BookOpen, 
  FileText, 
  Search, 
  Gavel, 
  MessageSquare, 
  ChevronRight, 
  Send, 
  User, 
  Briefcase, 
  LogOut, 
  ShieldCheck,
  Loader2,
  Menu,
  X,
  Upload,
  Mic,
  MicOff,
  Square,
  Download,
  FileUp,
  Plus,
  Trash2,
  Play,
  Pause,
  Languages,
  History,
  Type,
  TrendingUp,
  AlertCircle,
  Clock,
  CheckCircle2,
  Info,
  Copy,
  ExternalLink
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import ReactMarkdown from 'react-markdown';
import { jsPDF } from 'jspdf';
import { cn } from './lib/utils';
import { getLegalAdvice, transcribeAudio, summarizeDocument, ask_question, processCasePDF, recommendCases, verifyCitation } from './services/gemini';
import { UserRole, Message, LawyerFeature, Session, CaseRecord, RecommendationResult, CitationVerificationResult } from './types';

const LandingPage = ({ onSelectRole }: { onSelectRole: (role: UserRole) => void }) => {
  const [activeSection, setActiveSection] = useState('home');
  const [selectedArticle, setSelectedArticle] = useState<typeof articles[0] | null>(null);

  const solutions = [
    {
      id: 'judiciary',
      title: 'Judiciary',
      description: 'Advanced case management and prioritization for judicial officers.',
      features: ['Case Classifier', 'Priority Engine', 'Recommendation Engine'],
      role: 'deskaid' as UserRole
    },
    {
      id: 'lawyers',
      title: 'Lawyers',
      description: 'Elite drafting, research, and analysis tools for legal professionals.',
      features: ['Legal Drafting', 'Case Research', 'Document Analysis'],
      role: 'lawyer' as UserRole
    },
    {
      id: 'public',
      title: 'Public',
      description: 'Accessible legal guidance and law understanding for the general public.',
      features: ['Legal Chatbot', 'Law Understanding', 'Guidance Tools'],
      role: 'user' as UserRole
    }
  ];

  const articles = [
    {
      title: "Why Legal AI is No Longer Optional",
      excerpt: "As case loads increase and inefficiencies mount, AI is becoming a critical tool for improving legal outcomes.",
      content: "The modern legal landscape is characterized by an overwhelming volume of data and an ever-increasing backlog of cases. Traditional manual workflows are no longer sufficient to meet the demands of a fast-paced society. Legal AI systems, like Verdict AI, provide the necessary intelligence to process information at scale, identify patterns that humans might miss, and significantly reduce the time spent on routine tasks. This isn't about replacing legal professionals; it's about empowering them with the tools they need to deliver justice more effectively."
    },
    {
      title: "The Problem with Traditional Legal Workflows",
      excerpt: "Manual reading, delays, and a lack of prioritization are hindering the efficiency of legal systems.",
      content: "Legal professionals often spend up to 40% of their time on research and document review. This manual labor is not only time-consuming but also prone to human error. Furthermore, the lack of automated prioritization means that critical cases may languish while less urgent matters are addressed. Verdict AI addresses these bottlenecks by automating document analysis and providing a data-driven priority engine, ensuring that resources are allocated where they are needed most."
    },
    {
      title: "How AI Can Assist — Not Replace — Legal Professionals",
      excerpt: "Emphasizing trust and human-AI collaboration in the legal field.",
      content: "The goal of Verdict AI is to augment human intelligence, not replace it. Legal work requires a level of nuance, empathy, and ethical judgment that AI cannot replicate. However, AI can handle the heavy lifting of data processing, citation verification, and initial drafting. By offloading these tasks to a reliable system, lawyers and judges can focus on the high-level strategic and ethical dimensions of their work. Trust is built through transparency and explainability, which are core pillars of our system."
    }
  ];

  return (
    <div className="min-h-screen bg-brand-dark text-slate-300 font-sans selection:bg-brand-blue selection:text-white">
      {/* Article Modal */}
      <AnimatePresence>
        {selectedArticle && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-8"
          >
            <div className="absolute inset-0 bg-black/90 backdrop-blur-sm" onClick={() => setSelectedArticle(null)} />
            <motion.div 
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="relative w-full max-w-4xl bg-brand-gray border border-white/10 rounded-3xl overflow-hidden shadow-2xl flex flex-col max-h-full"
            >
              <button 
                onClick={() => setSelectedArticle(null)}
                className="absolute top-6 right-6 p-2 bg-black/50 hover:bg-black/80 text-white rounded-full transition-all z-10"
              >
                <X className="w-5 h-5" />
              </button>
              
              <div className="overflow-y-auto custom-scrollbar">
                <div className="aspect-video w-full overflow-hidden">
                  <img 
                    src={`https://picsum.photos/seed/${selectedArticle.title}/1200/675`} 
                    alt={selectedArticle.title} 
                    className="w-full h-full object-cover opacity-80"
                    referrerPolicy="no-referrer"
                  />
                </div>
                <div className="p-10 sm:p-16">
                  <div className="flex items-center gap-4 mb-6">
                    <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-brand-blue px-3 py-1 bg-brand-blue/10 rounded-full border border-brand-blue/20">
                      Legal Intelligence
                    </span>
                    <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                      5 Min Read
                    </span>
                  </div>
                  <h2 className="text-4xl sm:text-5xl font-display font-bold text-white mb-8 tracking-tight leading-tight">
                    {selectedArticle.title}
                  </h2>
                  <div className="prose prose-invert max-w-none">
                    <p className="text-xl text-slate-400 leading-relaxed mb-8 font-medium italic">
                      {selectedArticle.excerpt}
                    </p>
                    <div className="space-y-6 text-slate-300 leading-relaxed text-lg">
                      <p>
                        The legal landscape is undergoing a profound transformation. As complexity increases and the volume of data grows exponentially, traditional methods of legal research and case management are being challenged. Verdict AI represents a new paradigm in legal intelligence—one that prioritizes precision, authority, and efficiency.
                      </p>
                      <p>
                        Our system is built on the principle that legal technology should not just automate tasks, but enhance human judgment. By leveraging advanced language models and structured legal datasets, we provide practitioners and the judiciary with the tools they need to navigate the complexities of modern law.
                      </p>
                      <h3 className="text-2xl font-bold text-white mt-12 mb-4">The Future of Precedent</h3>
                      <p>
                        One of the most critical challenges in the Pakistani legal system is the identification and application of relevant precedents. With thousands of judgments delivered annually, finding the "needle in the haystack" is increasingly difficult. Verdict AI's recommendation engine uses semantic search and legal reasoning to identify cases that are not just keyword-matched, but legally relevant.
                      </p>
                      <p>
                        This ensures that the judiciary can make informed decisions based on the most accurate and up-to-date case law, while lawyers can build stronger arguments with less effort.
                      </p>
                      <h3 className="text-2xl font-bold text-white mt-12 mb-4">Explainability and Trust</h3>
                      <p>
                        In law, a conclusion is only as good as the reasoning behind it. That's why Verdict AI is designed with "Explainability First." Every AI-generated output includes the underlying reasoning, confidence levels, and direct references to statutes or case law. This transparency builds trust and ensures that the system remains a tool for professionals, not a replacement for them.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 h-20 border-b border-white/5 bg-brand-dark/80 backdrop-blur-md z-50 px-8 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white rounded flex items-center justify-center">
            <Scale className="w-5 h-5 text-black" />
          </div>
          <span className="font-display font-bold text-xl tracking-tight text-white">Verdict AI</span>
        </div>
        <div className="hidden md:flex items-center gap-10">
          <button onClick={() => document.getElementById('solutions')?.scrollIntoView({ behavior: 'smooth' })} className="text-sm font-medium hover:text-white transition-colors">Solutions</button>
          <button onClick={() => document.getElementById('tools')?.scrollIntoView({ behavior: 'smooth' })} className="text-sm font-medium hover:text-white transition-colors">Tools</button>
          <button onClick={() => document.getElementById('articles')?.scrollIntoView({ behavior: 'smooth' })} className="text-sm font-medium hover:text-white transition-colors">Articles</button>
          <button onClick={() => document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' })} className="text-sm font-medium hover:text-white transition-colors">About</button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-40 pb-20 px-8">
        <div className="max-w-4xl mx-auto text-center">
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-6xl md:text-7xl font-display font-bold text-white mb-8 tracking-tight"
          >
            Legal intelligence, redefined.
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed"
          >
            An elite, enterprise-grade legal intelligence platform designed to improve how legal systems function for the Judiciary, Lawyers, and the Public.
          </motion.p>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <button onClick={() => document.getElementById('solutions')?.scrollIntoView({ behavior: 'smooth' })} className="btn-primary w-full sm:w-auto px-8">
              Explore Solutions
            </button>
            <button onClick={() => document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' })} className="btn-secondary w-full sm:w-auto px-8">
              Our Mission
            </button>
          </motion.div>
        </div>
      </section>

      {/* Solutions Section */}
      <section id="solutions" className="py-32 px-8 border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <div className="mb-20 text-center">
            <h2 className="text-4xl font-bold mb-4">Solutions</h2>
            <p className="text-slate-500">Tailored intelligence for every pillar of the legal system.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {solutions.map((sol, idx) => (
              <motion.div 
                key={sol.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="glass-panel p-10 flex flex-col group hover:border-white/20 transition-all cursor-pointer"
                onClick={() => onSelectRole(sol.role)}
              >
                <h3 className="text-2xl font-bold mb-4">{sol.title}</h3>
                <p className="text-slate-500 mb-8 text-sm leading-relaxed">{sol.description}</p>
                <div className="mt-auto space-y-3">
                  {sol.features.map(f => (
                    <div key={f} className="flex items-center gap-3 text-xs font-medium text-slate-400">
                      <div className="w-1 h-1 bg-brand-blue rounded-full" />
                      {f}
                    </div>
                  ))}
                </div>
                <div className="mt-10 pt-6 border-t border-white/5 flex items-center justify-between group-hover:text-white transition-colors">
                  <span className="text-xs font-bold uppercase tracking-widest">Enter Desk</span>
                  <ChevronRight className="w-4 h-4" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Tools Section */}
      <section id="tools" className="py-32 px-8 bg-white/[0.01] border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-20 items-center">
            <div>
              <h2 className="text-4xl font-bold mb-6">A Unified Intelligence System</h2>
              <p className="text-slate-400 mb-10 leading-relaxed">
                Verdict AI isn't just a collection of tools. It's a unified system where every module works in harmony to provide a seamless legal workflow.
              </p>
              <div className="grid grid-cols-2 gap-6">
                {[
                  { name: 'Legal Chatbot', icon: MessageSquare },
                  { name: 'Drafting Engine', icon: FileText },
                  { name: 'Summarizer', icon: BookOpen },
                  { name: 'Priority Engine', icon: TrendingUp },
                  { name: 'Case Law Search', icon: Search },
                  { name: 'Law Understanding', icon: Gavel }
                ].map(tool => (
                  <div key={tool.name} className="flex items-center gap-3 p-4 rounded-xl bg-white/5 border border-white/5">
                    <tool.icon className="w-4 h-4 text-brand-blue" />
                    <span className="text-xs font-medium text-slate-300">{tool.name}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="relative">
              <div className="aspect-square glass-panel p-8 flex flex-col gap-6 shadow-2xl">
                <div className="h-12 w-full bg-white/5 rounded-lg animate-pulse" />
                <div className="flex-1 w-full bg-white/5 rounded-lg animate-pulse" />
                <div className="h-24 w-full bg-white/5 rounded-lg animate-pulse" />
              </div>
              <div className="absolute -bottom-6 -right-6 w-48 h-48 bg-brand-blue/20 blur-3xl rounded-full" />
            </div>
          </div>
        </div>
      </section>

      {/* Articles Section */}
      <section id="articles" className="py-32 px-8 border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <div className="mb-20">
            <h2 className="text-4xl font-bold mb-4">Insights</h2>
            <p className="text-slate-500">Perspectives on the future of legal intelligence.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-12">
            {articles.map((art, idx) => (
              <motion.article 
                key={art.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                className="flex flex-col group"
              >
                <div 
                  className="aspect-video bg-white/5 rounded-2xl mb-6 overflow-hidden cursor-pointer"
                  onClick={() => setSelectedArticle(art)}
                >
                  <img src={`https://picsum.photos/seed/${art.title}/800/450`} alt={art.title} className="w-full h-full object-cover opacity-50 group-hover:opacity-80 group-hover:scale-105 transition-all duration-500" referrerPolicy="no-referrer" />
                </div>
                <h3 
                  className="text-xl font-bold mb-4 hover:text-brand-blue transition-colors cursor-pointer leading-tight"
                  onClick={() => setSelectedArticle(art)}
                >
                  {art.title}
                </h3>
                <p className="text-slate-500 text-sm leading-relaxed mb-6 line-clamp-3">{art.excerpt}</p>
                <button 
                  onClick={() => setSelectedArticle(art)}
                  className="text-xs font-bold uppercase tracking-widest text-white flex items-center gap-2 hover:gap-3 transition-all"
                >
                  Read Article <ChevronRight className="w-4 h-4 text-brand-blue" />
                </button>
              </motion.article>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-32 px-8 border-t border-white/5 bg-white/[0.01]">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-8">Our Mission</h2>
          <p className="text-xl text-slate-400 leading-relaxed mb-12">
            Verdict AI exists to bring clarity to legal complexity. We believe that justice should be accessible, efficient, and fair. By building high-authority intelligence systems, we empower the judiciary, lawyers, and the public to navigate the legal landscape with confidence.
          </p>
          <div className="flex items-center justify-center gap-12">
            <div>
              <div className="text-3xl font-bold text-white mb-1">24/7</div>
              <div className="text-xs text-slate-500 uppercase tracking-widest font-bold">System Availability</div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-20 px-8 border-t border-white/5">
        <div className="max-w-6xl mx-auto flex flex-col md:row items-center justify-between gap-8">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 bg-white rounded flex items-center justify-center">
              <Scale className="w-4 h-4 text-black" />
            </div>
            <span className="font-display font-bold tracking-tight text-white">Verdict AI</span>
          </div>
          <div className="text-[10px] text-slate-600 uppercase tracking-[0.2em]">
            © 2026 Verdict AI • Professional Legal Intelligence System
          </div>
          <div className="flex gap-8">
            <button className="text-[10px] font-bold uppercase tracking-widest text-slate-500 hover:text-white transition-colors">Privacy</button>
            <button className="text-[10px] font-bold uppercase tracking-widest text-slate-500 hover:text-white transition-colors">Terms</button>
            <button className="text-[10px] font-bold uppercase tracking-widest text-slate-500 hover:text-white transition-colors">Contact</button>
          </div>
        </div>
      </footer>
    </div>
  );
};

// --- Components ---

const AuthBreak = ({ role, onComplete }: { role: UserRole, onComplete: () => void }) => {
  const [loading, setLoading] = useState(false);

  const handleAuth = async () => {
    setLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    setLoading(false);
    onComplete();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950 p-6">
      <motion.div 
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="max-w-md w-full bg-slate-900 p-8 rounded-2xl shadow-2xl border border-slate-800 text-center"
      >
        <div className="w-16 h-16 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
          <ShieldCheck className="w-8 h-8 text-emerald-500" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">Authorization Break</h2>
        <p className="text-slate-400 mb-8">
          Verifying credentials for Pakistani {role === 'lawyer' ? 'Legal Professional' : role === 'deskaid' ? 'Court Transcriptionist' : 'General User'} access...
        </p>
        <button 
          onClick={handleAuth}
          disabled={loading}
          className="w-full py-4 px-6 bg-emerald-600 text-white rounded-xl font-bold flex items-center justify-center gap-3 hover:bg-emerald-700 transition-all disabled:opacity-50"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Authorize Access'}
        </button>
      </motion.div>
    </div>
  );
};

const ChatInterface = ({ 
  role, 
  feature, 
  onBack 
}: { 
  role: UserRole, 
  feature?: LawyerFeature, 
  onBack?: () => void 
}) => {
  const getInitialMessage = () => ({
    id: '1',
    role: 'assistant' as const,
    content: role === 'user' 
      ? "I am Verdict AI. How can I assist with your legal inquiries today? I can provide advice, explain jargon, and search for Pakistani case law."
      : `Verdict AI Professional Suite: ${feature ? feature.charAt(0).toUpperCase() + feature.slice(1).replace('-', ' ') : 'General Chat'} active. How can I assist your practice?`,
    timestamp: Date.now()
  });

  const [messages, setMessages] = useState<Message[]>([getInitialMessage()]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<{ name: string, content: string, file?: File } | null>(null);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset messages when feature changes
  useEffect(() => {
    setMessages([getInitialMessage()]);
  }, [feature, role]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedFile({
          name: file.name,
          content: event.target?.result as string,
          file: file
        });
      };
      reader.readAsText(file);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !uploadedFile) return;
    if (loading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input + (uploadedFile ? `\n\n[Attached File: ${uploadedFile.name}]` : ''),
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    let response = "";
    if (feature === 'summarizer' && uploadedFile?.file) {
      const result = await summarizeDocument(uploadedFile.file);
      response = result.summary;
      if (result.fileId) setActiveFileId(result.fileId);
    } else if (feature === 'summarizer' && activeFileId) {
      // If we are in summarizer mode and have an active file, ask a question about it
      const result = await ask_question(activeFileId, input);
      response = result.answer;
    } else {
      response = await getLegalAdvice(input, role === 'lawyer', feature, uploadedFile?.content);
    }
    
    const assistantMsg: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response || "I'm sorry, I couldn't process that.",
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, assistantMsg]);
    setLoading(false);
    setUploadedFile(null);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const downloadMessageAsPDF = (content: string, id: string) => {
    const doc = new jsPDF();
    const splitText = doc.splitTextToSize(content, 180);
    doc.text(splitText, 10, 10);
    doc.save(`LegumAI_Message_${id}.pdf`);
  };

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] font-sans text-slate-300">
      <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl mx-auto w-full" ref={scrollRef}>
        <div className="bg-brand-blue/5 border border-brand-blue/10 p-4 rounded-lg text-brand-blue/80 text-xs mb-8 flex items-center gap-3">
          <ShieldCheck className="w-4 h-4 shrink-0" />
          <span>Verdict AI is strictly limited to Pakistani Law. All responses are backed by High Court/Supreme Court references.</span>
        </div>

        {feature === 'summarizer' && messages.length === 1 && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/[0.02] border border-white/5 rounded-2xl p-8 text-center border-dashed"
          >
            <div className="w-12 h-12 bg-blue-500/10 rounded-xl flex items-center justify-center mx-auto mb-4 border border-blue-500/20">
              <FileText className="w-6 h-6 text-blue-500" />
            </div>
            <h3 className="text-white font-bold mb-2">Verdict Summarizer</h3>
            <p className="text-sm text-slate-500 mb-6 max-w-sm mx-auto">
              Upload a court verdict or legal PDF to get an instant, concise summary of the judgment.
            </p>
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="px-6 py-3 bg-blue-500 text-white rounded-xl font-bold text-xs uppercase tracking-widest hover:bg-blue-600 transition-all"
            >
              Upload Verdict PDF
            </button>
          </motion.div>
        )}

        {feature === 'drafting' && messages.length === 1 && !uploadedFile && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/[0.02] border border-white/5 rounded-2xl p-8 text-center border-dashed"
          >
            <div className="w-12 h-12 bg-blue-500/10 rounded-xl flex items-center justify-center mx-auto mb-4 border border-blue-500/20">
              <FileUp className="w-6 h-6 text-blue-500" />
            </div>
            <h3 className="text-white font-bold mb-2">Reference Document</h3>
            <p className="text-sm text-slate-500 mb-6 max-w-sm mx-auto">
              Upload an existing legal document, contract, or draft to use as a template or reference for your new work.
            </p>
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="px-6 py-3 bg-white/5 text-white rounded-xl font-bold text-xs uppercase tracking-widest border border-white/10 hover:bg-white/10 transition-all"
            >
              Upload Reference
            </button>
          </motion.div>
        )}

        {messages.map((msg) => (
          <div 
            key={msg.id}
            className={cn(
              "flex w-full animate-in fade-in slide-in-from-bottom-2 duration-300",
              msg.role === 'user' ? "justify-end" : "justify-start"
            )}
          >
            <div className={cn(
              "max-w-[85%] p-5 rounded-2xl shadow-sm",
              msg.role === 'user' 
                ? "bg-brand-blue text-white ml-auto rounded-tr-none" 
                : "bg-brand-gray border border-white/5 text-slate-200 mr-auto rounded-tl-none"
            )}>
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              
              {msg.role === 'assistant' && (
                <div className="mt-6 pt-4 border-t border-white/5 space-y-4">
                  <div className="flex items-center justify-end">
                    <div className="flex items-center gap-2">
                      <button 
                        onClick={() => copyToClipboard(msg.content)}
                        className="p-1.5 hover:bg-white/5 rounded text-slate-500 hover:text-white transition-colors" 
                        title="Copy"
                      >
                        <Copy className="w-3 h-3" />
                      </button>
                      <button 
                        onClick={() => downloadMessageAsPDF(msg.content, msg.id)}
                        className="p-1.5 hover:bg-white/5 rounded text-slate-500 hover:text-white transition-colors" 
                        title="Export PDF"
                      >
                        <Download className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                  
                  <details className="group">
                    <summary className="text-[10px] font-bold uppercase tracking-widest text-slate-500 cursor-pointer hover:text-slate-300 transition-colors flex items-center gap-2 list-none">
                      <ChevronRight className="w-3 h-3 group-open:rotate-90 transition-transform" />
                      AI Reasoning & References
                    </summary>
                    <div className="mt-3 p-3 bg-black/20 rounded-lg text-[11px] text-slate-400 leading-relaxed border border-white/5">
                      <p className="mb-2"><strong>Basis:</strong> Analysis of Supreme Court precedents regarding civil procedure and constitutional rights.</p>
                      <p><strong>References:</strong> PLD 2021 SC 456, 2019 SCMR 789. The logic follows the principle of natural justice as interpreted in recent appellate rulings.</p>
                    </div>
                  </details>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-transparent border-l-2 border-blue-500/30 pl-6 p-4 flex gap-2">
              <div className="w-1.5 h-1.5 bg-blue-500/50 rounded-full animate-pulse" />
              <div className="w-1.5 h-1.5 bg-blue-500/50 rounded-full animate-pulse delay-75" />
              <div className="w-1.5 h-1.5 bg-blue-500/50 rounded-full animate-pulse delay-150" />
            </div>
          </div>
        )}
      </div>

      <div className="p-6 border-t border-white/5 bg-[#0a0a0a]">
        {uploadedFile && (
          <div className="max-w-4xl mx-auto mb-4 flex items-center justify-between bg-blue-500/5 border border-blue-500/10 p-3 rounded-lg">
            <div className="flex items-center gap-3 text-sm text-blue-400">
              <FileUp className="w-4 h-4" />
              <span className="truncate max-w-[300px] font-medium">{uploadedFile.name}</span>
            </div>
            <button onClick={() => setUploadedFile(null)} className="text-slate-500 hover:text-white transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}
        <div className="max-w-4xl mx-auto flex gap-4 items-end">
          <div className="flex-1 relative group">
            <textarea 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask Verdict AI..."
              className="w-full bg-white/5 border border-white/10 rounded-xl px-5 py-4 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all resize-none min-h-[56px] max-h-[300px] text-slate-200 placeholder:text-slate-600"
              rows={1}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = `${target.scrollHeight}px`;
              }}
            />
          </div>
          
          <div className="flex gap-2">
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="bg-white/5 border border-white/10 text-slate-400 p-4 rounded-xl hover:bg-white/10 hover:text-white transition-all"
              title="Upload Document"
            >
              <Upload className="w-5 h-5" />
            </button>
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileUpload} 
              className="hidden" 
              accept=".pdf,.txt,.doc,.docx"
            />
            <button 
              onClick={handleSend}
              disabled={loading || (!input.trim() && !uploadedFile)}
              className="bg-blue-600 text-white p-4 rounded-xl hover:bg-blue-500 transition-all disabled:opacity-20 disabled:grayscale shadow-lg shadow-blue-600/20"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
        <div className="max-w-4xl mx-auto mt-4 text-center">
          <p className="text-[10px] text-slate-600 uppercase tracking-[0.2em]">Verdict AI • Professional Legal Intelligence • Pakistan</p>
        </div>
      </div>
    </div>
  );
};

const CaseWorkspace = ({ 
  activeFile, 
  analysisResult, 
  onFileUpload, 
  onAnalyze,
  isProcessing 
}: { 
  activeFile: File | null, 
  analysisResult: RecommendationResult | null, 
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void,
  onAnalyze: () => void,
  isProcessing: boolean
}) => {
  return (
    <div className="flex h-full overflow-hidden">
      {/* Left: Case Files */}
      <div className="w-80 border-r border-white/5 flex flex-col bg-brand-gray/50">
        <div className="p-6 border-b border-white/5">
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Case Documents</h3>
          <label className="btn-secondary w-full cursor-pointer text-xs">
            <Upload className="w-3 h-3" /> Upload File
            <input type="file" className="hidden" onChange={onFileUpload} />
          </label>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {activeFile ? (
            <div className="p-3 bg-white/5 rounded-lg border border-white/10 flex items-center gap-3">
              <FileText className="w-4 h-4 text-brand-blue" />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium text-white truncate">{activeFile.name}</div>
                <div className="text-[10px] text-slate-500">{(activeFile.size / 1024).toFixed(1)} KB</div>
              </div>
            </div>
          ) : (
            <div className="h-40 flex flex-col items-center justify-center text-slate-600 text-center px-4">
              <FileUp className="w-8 h-8 mb-2 opacity-20" />
              <p className="text-[10px] font-medium uppercase tracking-widest">No documents uploaded</p>
            </div>
          )}
        </div>
      </div>

      {/* Center: AI Analysis */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="p-6 border-b border-white/5 flex items-center justify-between bg-brand-dark">
          <h2 className="text-xl font-bold">Intelligence Analysis</h2>
          {activeFile && !analysisResult && !isProcessing && (
            <button onClick={onAnalyze} className="btn-primary py-2 text-xs">
              Run Analysis
            </button>
          )}
        </div>
        <div className="flex-1 overflow-y-auto p-8 custom-scrollbar">
          {isProcessing ? (
            <div className="h-full flex flex-col items-center justify-center gap-4">
              <Loader2 className="w-10 h-10 text-brand-blue animate-spin" />
              <p className="text-sm font-medium animate-pulse">Processing legal intelligence...</p>
            </div>
          ) : analysisResult ? (
            <div className="space-y-8 max-w-3xl mx-auto">
              <div className="grid grid-cols-2 gap-6">
                <div className="glass-panel p-6">
                  <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-2">Case Type</div>
                  <div className="text-lg font-bold text-white">{analysisResult.detected_type}</div>
                </div>
                <div className="glass-panel p-6">
                  <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-2">Risk Level</div>
                  <div className={cn(
                    "text-lg font-bold",
                    analysisResult.risk_level === 'High' ? 'text-red-500' : analysisResult.risk_level === 'Medium' ? 'text-amber-500' : 'text-emerald-500'
                  )}>{analysisResult.risk_level}</div>
                </div>
              </div>

              <div className="glass-panel p-6">
                <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-4">Likely Outcome</div>
                <p className="text-slate-300 leading-relaxed">{analysisResult.likely_outcome}</p>
              </div>

              <div className="space-y-4">
                <h3 className="text-sm font-bold uppercase tracking-widest text-slate-500">Similar Precedents</h3>
                {analysisResult.similar_cases.map((c, idx) => (
                  <div key={idx} className="glass-panel p-6 hover:border-white/10 transition-all">
                    <div className="flex justify-between items-start mb-4">
                      <h4 className="font-bold text-white">{c.title}</h4>
                      <span className="text-brand-blue font-bold">approximately {c.similarity}%</span>
                    </div>
                    <p className="text-xs text-slate-400 mb-4">{c.relevance_reason}</p>
                    <div className="flex items-center gap-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                      <span>{c.citation}</span>
                      <span>{c.court}</span>
                      <span>{c.year}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-600">
              <div className="w-20 h-20 glass-panel flex items-center justify-center mb-6">
                <ShieldCheck className="w-10 h-10 opacity-20" />
              </div>
              <p className="text-sm font-medium uppercase tracking-widest">Upload and analyze to see results</p>
            </div>
          )}
        </div>
      </div>

      {/* Right: AI Assistant */}
      <div className="w-96 border-l border-white/5 flex flex-col bg-brand-gray/50">
        <div className="p-6 border-b border-white/5">
          <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500">Assistant & Tools</h3>
        </div>
        <div className="flex-1 overflow-hidden">
          <ChatInterface role="lawyer" />
        </div>
      </div>
    </div>
  );
};

const LawyerDashboard = ({ onLogout }: { onLogout: () => void }) => {
  const [activeFeature, setActiveFeature] = useState<LawyerFeature | 'case-workspace'>('chat');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeFile, setActiveFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<RecommendationResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const features = [
    { id: 'chat', label: 'Intelligence Chat', icon: MessageSquare },
    { id: 'drafting', label: 'Legal Drafting', icon: FileText },
    { id: 'summarizer', label: 'Verdict Summarizer', icon: FileText },
    { id: 'case-recommendation', label: 'Case Recommendation', icon: BookOpen },
    { id: 'case-law', label: 'Case Law Search', icon: Search },
  ];

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setActiveFile(file);
      setAnalysisResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!activeFile) return;
    setIsProcessing(true);
    try {
      const caseRecord = await processCasePDF(activeFile);
      const res = await recommendCases(caseRecord.signals.summary || activeFile.name);
      setAnalysisResult(res);
    } catch (error) {
      console.error("Analysis Error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen bg-brand-dark overflow-hidden font-sans">
      <motion.aside 
        initial={false}
        animate={{ width: sidebarOpen ? 280 : 80 }}
        className="bg-brand-gray border-r border-white/5 text-white flex flex-col transition-all duration-300 z-20"
      >
        <div className="p-8 flex items-center justify-between">
          {sidebarOpen && (
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-white rounded flex items-center justify-center">
                <Scale className="w-4 h-4 text-black" />
              </div>
              <span className="font-display font-bold tracking-tight text-white">Verdict AI</span>
            </div>
          )}
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-white/5 rounded-lg text-slate-400">
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        <nav className="flex-1 px-4 space-y-1 mt-4 overflow-y-auto custom-scrollbar">
          {features.map((f) => (
            <button
              key={f.id}
              onClick={() => setActiveFeature(f.id as LawyerFeature | 'case-workspace')}
              className={cn(
                "w-full flex items-center gap-4 p-3 rounded-xl transition-all group",
                activeFeature === f.id ? "bg-white/5 text-white" : "text-slate-500 hover:text-slate-300 hover:bg-white/[0.02]"
              )}
            >
              <f.icon className={cn("w-5 h-5 shrink-0", activeFeature === f.id ? "text-brand-blue" : "group-hover:text-brand-blue")} />
              {sidebarOpen && <span className="text-sm font-medium">{f.label}</span>}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-white/5">
          <button 
            onClick={onLogout}
            className="w-full flex items-center gap-4 p-3 text-slate-500 hover:text-red-400 hover:bg-red-400/10 rounded-xl transition-all"
          >
            <LogOut className="w-5 h-5 shrink-0" />
            {sidebarOpen && <span className="text-sm font-medium">Exit System</span>}
          </button>
        </div>
      </motion.aside>

      <main className="flex-1 flex flex-col min-w-0">
        <header className="h-16 bg-brand-dark border-b border-white/5 flex items-center justify-between px-8 shrink-0">
          <div className="flex items-center gap-4">
            <h2 className="font-bold text-white tracking-tight uppercase text-xs tracking-[0.2em]">
              {features.find(f => f.id === activeFeature)?.label || 'System'}
            </h2>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right hidden sm:block">
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">System Status</p>
              <p className="text-xs font-bold text-emerald-500">Operational</p>
            </div>
            <div className="w-8 h-8 bg-white/5 rounded-full flex items-center justify-center border border-white/10">
              <User className="w-4 h-4 text-slate-400" />
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-hidden">
          {activeFeature === 'priority-engine' ? (
            <PriorityEngine />
          ) : activeFeature === 'case-recommendation' ? (
            <CaseRecommendation />
          ) : activeFeature === 'citation-verification' ? (
            <CitationVerification />
          ) : (
            <ChatInterface role="lawyer" feature={activeFeature as LawyerFeature} />
          )}
        </div>
      </main>
    </div>
  );
};

const DeskAidDashboard = ({ onLogout }: { onLogout: () => void }) => {
  const [activeFeature, setActiveFeature] = useState<'classifier' | 'priority' | 'recommendation' | 'transcription'>('classifier');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const features = [
    { id: 'classifier', label: 'Case Classifier (Coming Soon)', icon: ShieldCheck, disabled: true },
    { id: 'priority', label: 'Priority Engine', icon: TrendingUp },
    { id: 'recommendation', label: 'Recommendation Engine', icon: BookOpen },
    { id: 'transcription', label: 'Court Transcription', icon: Mic },
  ];

  return (
    <div className="flex h-screen bg-brand-dark overflow-hidden font-sans text-slate-300">
      <motion.aside 
        initial={false}
        animate={{ width: sidebarOpen ? 280 : 80 }}
        className="bg-brand-gray border-r border-white/5 text-white flex flex-col transition-all duration-300 z-20"
      >
        <div className="p-8 flex items-center justify-between">
          {sidebarOpen && (
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-white rounded flex items-center justify-center">
                <Scale className="w-4 h-4 text-black" />
              </div>
              <span className="font-display font-bold tracking-tight text-white">Verdict AI</span>
            </div>
          )}
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-white/5 rounded-lg text-slate-400">
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        <nav className="flex-1 px-4 space-y-1 mt-4 overflow-y-auto custom-scrollbar">
          {features.map((f) => (
            <button
              key={f.id}
              disabled={(f as any).disabled}
              onClick={() => setActiveFeature(f.id as any)}
              className={cn(
                "w-full flex items-center gap-4 p-3 rounded-xl transition-all group",
                activeFeature === f.id ? "bg-white/5 text-white" : "text-slate-500 hover:text-slate-300 hover:bg-white/[0.02]",
                (f as any).disabled && "opacity-50 cursor-not-allowed hover:bg-transparent"
              )}
            >
              <f.icon className={cn("w-5 h-5 shrink-0", activeFeature === f.id ? "text-brand-blue" : "group-hover:text-brand-blue")} />
              {sidebarOpen && <span className="text-sm font-medium">{f.label}</span>}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-white/5">
          <button 
            onClick={onLogout}
            className="w-full flex items-center gap-4 p-3 text-slate-500 hover:text-red-400 hover:bg-red-400/10 rounded-xl transition-all"
          >
            <LogOut className="w-5 h-5 shrink-0" />
            {sidebarOpen && <span className="text-sm font-medium">Exit System</span>}
          </button>
        </div>
      </motion.aside>

      <main className="flex-1 flex flex-col min-w-0">
        <header className="h-16 bg-brand-dark border-b border-white/5 flex items-center justify-between px-8 shrink-0">
          <div className="flex items-center gap-4">
            <h2 className="font-bold text-white tracking-tight uppercase text-xs tracking-[0.2em]">
              Judiciary Suite • {features.find(f => f.id === activeFeature)?.label}
            </h2>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right hidden sm:block">
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Judicial Authority</p>
              <p className="text-xs font-bold text-brand-blue">Verified Access</p>
            </div>
            <div className="w-8 h-8 bg-white/5 rounded-full flex items-center justify-center border border-white/10">
              <User className="w-4 h-4 text-slate-400" />
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-hidden">
          {activeFeature === 'classifier' ? (
            <div className="p-10 max-w-4xl mx-auto h-full flex flex-col items-center justify-center text-center">
              <div className="w-24 h-24 bg-brand-blue/10 rounded-3xl flex items-center justify-center mb-8 border border-brand-blue/20">
                <ShieldCheck className="w-12 h-12 text-brand-blue" />
              </div>
              <h2 className="text-4xl font-black text-white mb-4 tracking-tight">Case Classifier</h2>
              <p className="text-xl text-slate-500 mb-8 max-w-lg">
                This module is currently under development to ensure 100% accuracy in legal categorization.
              </p>
              <div className="px-6 py-2 bg-white/5 border border-white/10 rounded-full text-brand-blue font-bold text-xs uppercase tracking-widest animate-pulse">
                Future Coming • Q3 2026
              </div>
            </div>
          ) : activeFeature === 'priority' ? (
            <PriorityEngine />
          ) : activeFeature === 'recommendation' ? (
            <CaseRecommendation />
          ) : (
            <TranscriptionInterface />
          )}
        </div>
      </main>
    </div>
  );
};

const TranscriptionView = ({ 
  session, 
  onUpdateSession, 
  languageMode, 
  setLanguageMode 
}: { 
  session: Session; 
  onUpdateSession: (messages: Message[]) => void;
  languageMode: 'urdu' | 'english';
  setLanguageMode: (mode: 'urdu' | 'english') => void;
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [session.messages]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await handleTranscription(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleTranscription = async (audioBlob: Blob) => {
    setIsProcessing(true);
    try {
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = async () => {
        const base64Audio = (reader.result as string).split(',')[1];
        const result = await transcribeAudio(base64Audio);
        const newMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: result.urdu || result.english || '',
          englishContent: result.english,
          urduContent: result.urdu,
          timestamp: Date.now()
        };
        onUpdateSession([...session.messages, newMessage]);
        setIsProcessing(false);
      };
    } catch (error) {
      console.error("Transcription Error:", error);
      setIsProcessing(false);
    }
  };

  const downloadPDF = (msg: Message) => {
    const doc = new jsPDF();
    doc.setFontSize(18);
    doc.text("Verdict AI - Official Transcription", 10, 20);
    doc.setFontSize(10);
    doc.text(`Date: ${new Date(msg.timestamp).toLocaleString()}`, 10, 30);
    doc.text("Verdict AI Watermark", 150, 10);
    
    const content = msg.englishContent || msg.content;
    const splitText = doc.splitTextToSize(content || "", 180);
    doc.text(splitText, 10, 45);
    doc.save(`Transcription_${msg.id}.pdf`);
  };

  return (
    <div className="h-full flex flex-col bg-brand-dark">
      <div className="flex-1 overflow-y-auto p-8 space-y-6 custom-scrollbar">
        {session.messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-slate-600 gap-6">
            <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center border border-white/10">
              <Mic className="w-10 h-10 opacity-20" />
            </div>
            <p className="text-xs font-bold uppercase tracking-widest">System Ready for Input</p>
          </div>
        )}
        
        {session.messages.map((msg) => (
          <motion.div 
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6 max-w-3xl mx-auto"
          >
            <div className="flex justify-between items-start mb-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-brand-blue/10 rounded-lg flex items-center justify-center border border-brand-blue/20">
                  <FileText className="w-4 h-4 text-brand-blue" />
                </div>
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <button 
                onClick={() => downloadPDF(msg)}
                className="p-2 hover:bg-white/5 rounded-lg text-slate-500 hover:text-white transition-all"
                title="Download Official PDF"
              >
                <Download className="w-4 h-4" />
              </button>
            </div>
            
            <div className="space-y-4">
              {msg.urduContent && (
                <div className="text-right">
                  <p className="text-xl leading-relaxed text-white font-medium" dir="rtl">{msg.urduContent}</p>
                </div>
              )}
              {msg.englishContent && (
                <div className="pt-4 border-t border-white/5">
                  <p className="text-sm leading-relaxed text-slate-400 italic">{msg.englishContent}</p>
                </div>
              )}
              {!msg.urduContent && !msg.englishContent && (
                <p className="text-slate-300 leading-relaxed">{msg.content}</p>
              )}
            </div>
          </motion.div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-8 border-t border-white/5 bg-brand-gray/50 backdrop-blur-xl">
        <div className="max-w-3xl mx-auto flex flex-col items-center gap-6">
          <div className="flex items-center gap-2 p-1 bg-white/5 rounded-xl border border-white/10">
            <button 
              onClick={() => setLanguageMode('english')}
              className={cn(
                "px-4 py-2 rounded-lg text-[10px] font-bold uppercase tracking-widest transition-all",
                languageMode === 'english' ? "bg-brand-blue text-white shadow-lg shadow-brand-blue/20" : "text-slate-500 hover:text-slate-300"
              )}
            >
              English
            </button>
            <button 
              onClick={() => setLanguageMode('urdu')}
              className={cn(
                "px-4 py-2 rounded-lg text-[10px] font-bold uppercase tracking-widest transition-all",
                languageMode === 'urdu' ? "bg-brand-blue text-white shadow-lg shadow-brand-blue/20" : "text-slate-500 hover:text-slate-300"
              )}
            >
              Urdu
            </button>
          </div>

          <div className="flex items-center gap-8">
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={isRecording ? stopRecording : startRecording}
              className={cn(
                "w-20 h-20 rounded-full flex items-center justify-center transition-all shadow-2xl relative",
                isRecording ? "bg-red-500 shadow-red-500/40" : "bg-brand-blue shadow-brand-blue/40"
              )}
            >
              {isRecording ? (
                <Square className="w-8 h-8 text-white" />
              ) : (
                <Mic className="w-8 h-8 text-white" />
              )}
              {isRecording && (
                <motion.div 
                  initial={{ scale: 1, opacity: 0.5 }}
                  animate={{ scale: 1.5, opacity: 0 }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                  className="absolute inset-0 bg-red-500 rounded-full"
                />
              )}
            </motion.button>
          </div>

          <p className="text-[10px] font-bold text-slate-600 uppercase tracking-[0.2em]">
            {isRecording ? "Recording Audio..." : isProcessing ? "Processing Intelligence..." : "System Ready for Input"}
          </p>
        </div>
      </div>
    </div>
  );
};

const TranscriptionInterface = () => {
  const [sessions, setSessions] = useState<Session[]>(() => {
    const saved = localStorage.getItem('verdictai_sessions');
    return saved ? JSON.parse(saved) : [];
  });
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [languageMode, setLanguageMode] = useState<'urdu' | 'english'>('english');

  useEffect(() => {
    localStorage.setItem('verdictai_sessions', JSON.stringify(sessions));
  }, [sessions]);

  const createNewSession = () => {
    const newSession: Session = {
      id: Date.now().toString(),
      name: 'New Statement',
      messages: [],
      createdAt: Date.now()
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
  };

  const deleteSession = (id: string) => {
    setSessions(sessions.filter(s => s.id !== id));
    if (activeSessionId === id) setActiveSessionId(null);
  };

  const updateSessionMessages = (messages: Message[]) => {
    setSessions(prev => prev.map(s => {
      if (s.id === activeSessionId) {
        let name = s.name;
        if (s.messages.length === 0 && messages.length > 0) {
          name = messages[0].content.slice(0, 20) + '...';
        }
        return { ...s, messages, name };
      }
      return s;
    }));
  };

  const activeSession = sessions.find(s => s.id === activeSessionId);

  return (
    <div className="flex h-full bg-brand-dark overflow-hidden font-sans">
      <motion.aside 
        initial={false}
        animate={{ width: sidebarOpen ? 300 : 0 }}
        className="bg-brand-gray border-r border-white/5 flex flex-col transition-all duration-300 z-20 overflow-hidden"
      >
        <div className="p-6 flex items-center justify-between shrink-0">
          <h1 className="font-bold text-xl text-white tracking-tight">Desk Aid</h1>
          <button onClick={() => setSidebarOpen(false)} className="p-1 hover:bg-white/5 rounded text-slate-400">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="px-4 mb-6">
          <button 
            onClick={createNewSession}
            className="w-full py-3 px-4 bg-brand-blue/10 text-brand-blue border border-brand-blue/20 rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-brand-blue hover:text-white transition-all"
          >
            <Plus className="w-4 h-4" /> NEW SESSION
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 space-y-2 pb-6 custom-scrollbar">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2 px-2">History</div>
          {sessions.map((s) => (
            <div 
              key={s.id}
              className={cn(
                "group flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all",
                activeSessionId === s.id ? "bg-white/5 text-white" : "text-slate-400 hover:bg-white/[0.02]"
              )}
              onClick={() => setActiveSessionId(s.id)}
            >
              <div className="flex items-center gap-3 overflow-hidden">
                <History className="w-4 h-4 shrink-0" />
                <span className="truncate text-sm font-medium">{s.name}</span>
              </div>
              <button 
                onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }}
                className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-500/20 hover:text-red-500 rounded-lg transition-all"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      </motion.aside>

      <main className="flex-1 flex flex-col min-w-0">
        <div className="flex-1 overflow-hidden">
          {activeSession ? (
            <TranscriptionView 
              session={activeSession} 
              onUpdateSession={updateSessionMessages}
              languageMode={languageMode}
              setLanguageMode={setLanguageMode}
            />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-4">
              <div className="w-20 h-20 bg-white/5 rounded-3xl flex items-center justify-center border border-white/10">
                <History className="w-10 h-10" />
              </div>
              <div className="text-center">
                <h3 className="text-white font-bold mb-1">No Active Session</h3>
                <p className="text-sm">Select a session from history or start a new one.</p>
              </div>
              <button 
                onClick={createNewSession}
                className="btn-primary mt-4 px-8"
              >
                START NEW SESSION
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

const PriorityEngine = () => {
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [search, setSearch] = useState('');

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsProcessing(true);
    const newCases: CaseRecord[] = [];

    for (let i = 0; i < files.length; i++) {
      try {
        const result = await processCasePDF(files[i]);
        newCases.push(result);
      } catch (error) {
        console.error(`Error processing ${files[i].name}:`, error);
      }
    }

    setCases(prev => [...newCases, ...prev].sort((a, b) => b.score - a.score));
    setIsProcessing(false);
  };

  const filteredCases = cases.filter(c => 
    c.title.toLowerCase().includes(search.toLowerCase()) ||
    c.signals.case_type?.toLowerCase().includes(search.toLowerCase()) ||
    c.signals.summary?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="h-full flex flex-col bg-brand-dark p-8 overflow-hidden">
      <div className="max-w-6xl mx-auto w-full flex flex-col h-full">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-10">
          <div>
            <h2 className="text-3xl font-bold text-white tracking-tight mb-2">Case Priority Engine</h2>
            <p className="text-slate-500 text-sm">Upload pending case PDFs — AI ranks them by urgency based on Pakistani legal standards.</p>
          </div>
          
          <div className="flex gap-4">
            <label className="btn-primary cursor-pointer text-xs px-6">
              <Upload className="w-4 h-4" />
              Upload PDFs
              <input type="file" multiple accept=".pdf" className="hidden" onChange={handleUpload} disabled={isProcessing} />
            </label>
            <button 
              onClick={() => setCases([])}
              className="btn-secondary text-xs px-6"
            >
              Clear All
            </button>
          </div>
        </div>

        <div className="relative mb-8">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
          <input 
            type="text" 
            placeholder="Search cases by title, type, or summary..." 
            className="w-full bg-brand-gray border border-white/5 rounded-2xl py-4 pl-12 pr-6 text-white focus:outline-none focus:border-brand-blue/50 transition-all"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
          {isProcessing && (
            <div className="glass-panel p-10 flex flex-col items-center justify-center gap-4">
              <Loader2 className="w-10 h-10 text-brand-blue animate-spin" />
              <p className="text-sm font-medium animate-pulse">Analyzing cases for priority...</p>
            </div>
          )}
          
          {filteredCases.length === 0 && !isProcessing && (
            <div className="h-60 flex flex-col items-center justify-center text-slate-600">
              <TrendingUp className="w-12 h-12 mb-4 opacity-10" />
              <p className="text-xs font-bold uppercase tracking-widest">No cases analyzed</p>
            </div>
          )}

          {filteredCases.map((c, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.05 }}
              className="glass-panel p-6 flex flex-col md:flex-row gap-6 items-start md:items-center hover:border-white/10 transition-all"
            >
              <div className="w-16 h-16 bg-white/5 rounded-2xl flex flex-col items-center justify-center border border-white/10 shrink-0">
                <div className="text-xl font-black text-white">{c.score}</div>
                <div className="text-[8px] text-slate-500 font-bold uppercase tracking-widest text-center">Approx. Score</div>
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-lg font-bold text-white truncate">{c.title}</h3>
                  <span className={cn(
                    "px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-widest border",
                    c.tag === 'Critical' ? 'text-red-500 bg-red-500/10 border-red-500/20' : 
                    c.tag === 'Medium' ? 'text-amber-500 bg-amber-500/10 border-amber-500/20' : 
                    'text-emerald-500 bg-emerald-500/10 border-emerald-500/20'
                  )}>
                    {c.tag} Priority
                  </span>
                </div>
                <p className="text-xs text-slate-400 line-clamp-2 mb-3 leading-relaxed">{c.signals.summary}</p>
                <div className="flex flex-wrap gap-4">
                  <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                    <FileText className="w-3 h-3 text-brand-blue" />
                    {c.signals.case_type}
                  </div>
                  <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                    <Clock className="w-3 h-3 text-brand-blue" />
                    {c.signals.days_waiting || 0} Days Pending
                  </div>
                </div>
              </div>

              <div className="flex gap-2 w-full md:w-auto">
                <button className="btn-secondary py-2 px-4 text-[10px] flex-1 md:flex-none">View Details</button>
                <button className="btn-primary py-2 px-4 text-[10px] flex-1 md:flex-none">Assign Judge</button>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

const CaseRecommendation = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState<RecommendationResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeFile, setActiveFile] = useState<File | null>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setActiveFile(file);
    setIsProcessing(true);
    try {
      const caseRecord = await processCasePDF(file);
      setText(caseRecord.signals.summary || '');
      const res = await recommendCases(caseRecord.signals.summary || file.name);
      setResult(res);
    } catch (error) {
      console.error("File Recommendation Error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const copyCitation = (citation: string) => {
    navigator.clipboard.writeText(citation);
  };

  const exportAnalysis = () => {
    if (!result) return;
    const doc = new jsPDF();
    doc.setFontSize(18);
    doc.text("Verdict AI - Case Analysis Report", 10, 20);
    doc.setFontSize(10);
    doc.text(`Type: ${result.detected_type}`, 10, 30);
    doc.text(`Risk Level: ${result.risk_level}`, 10, 35);
    doc.text(`Likely Outcome: ${result.likely_outcome}`, 10, 40);
    
    doc.setFontSize(14);
    doc.text("Similar Precedents:", 10, 55);
    
    let y = 65;
    result.similar_cases.forEach((c, i) => {
      if (y > 250) { doc.addPage(); y = 20; }
      doc.setFontSize(11);
      doc.text(`${i+1}. ${c.title}`, 10, y);
      doc.setFontSize(9);
      doc.text(`Citation: ${c.citation} | Similarity: approximately ${c.similarity}%`, 15, y + 5);
      doc.text(`Outcome: ${c.outcome}`, 15, y + 10);
      y += 25;
    });
    
    doc.save(`Case_Analysis_${Date.now()}.pdf`);
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setIsProcessing(true);
    try {
      const res = await recommendCases(text);
      setResult(res);
    } catch (error) {
      console.error("Recommendation Error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-brand-dark p-8 overflow-hidden">
      <div className="max-w-6xl mx-auto w-full flex flex-col h-full">
        <div className="mb-10 flex justify-between items-end">
          <div>
            <h2 className="text-3xl font-bold text-white tracking-tight mb-2">Recommendation Engine</h2>
            <p className="text-slate-500 text-sm">Input an FIR or case summary to find similar Supreme Court precedents and analyze legal risks.</p>
          </div>
          {result && (
            <div className="flex gap-3">
              <button 
                onClick={exportAnalysis}
                className="btn-primary text-xs px-6"
              >
                <Download className="w-3 h-3" /> Export Analysis
              </button>
              <button 
                onClick={() => {setResult(null); setText(''); setActiveFile(null);}}
                className="btn-secondary text-xs px-6"
              >
                New Analysis
              </button>
            </div>
          )}
        </div>

        <div className="grid md:grid-cols-3 gap-8 flex-1 overflow-hidden">
          <div className="md:col-span-1 flex flex-col gap-6">
            <div className="glass-panel p-6">
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <FileText className="w-4 h-4 text-brand-blue" />
                Input Source
              </h3>
              
              <div className="space-y-4">
                <label className="block">
                  <div className={cn(
                    "border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all",
                    activeFile ? "border-brand-blue/50 bg-brand-blue/5" : "border-white/5 hover:border-white/10"
                  )}>
                    <Upload className={cn("w-8 h-8 mx-auto mb-2", activeFile ? "text-brand-blue" : "text-slate-600")} />
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                      {activeFile ? activeFile.name : "Upload Case PDF"}
                    </p>
                    <input type="file" accept=".pdf" className="hidden" onChange={handleFileUpload} disabled={isProcessing} />
                  </div>
                </label>

                <div className="relative">
                  <div className="absolute inset-0 flex items-center"><span className="w-full border-t border-white/5"></span></div>
                  <div className="relative flex justify-center text-[10px] uppercase font-bold text-slate-600"><span className="bg-brand-gray px-2">or paste text</span></div>
                </div>

                <textarea 
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste FIR content or case summary here..."
                  className="w-full h-48 bg-black/20 border border-white/5 rounded-xl p-4 text-slate-300 text-sm focus:outline-none focus:border-brand-blue/50 transition-all resize-none"
                />
                
                <button 
                  onClick={handleAnalyze}
                  disabled={isProcessing || !text.trim()}
                  className="btn-primary w-full py-4 text-xs"
                >
                  {isProcessing ? (
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </div>
                  ) : "Find Precedents"}
                </button>
              </div>
            </div>

            {result && (
              <div className="glass-panel p-6">
                <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-amber-500" />
                  Risk Analysis
                </h3>
                <div className="space-y-4">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Detected Type</div>
                    <div className="text-white font-medium">{result.detected_type}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Risk Level</div>
                    <div className={cn(
                      "inline-block px-3 py-1 rounded-full text-[10px] font-bold tracking-widest uppercase",
                      result.risk_level === 'High' ? "bg-red-500/10 text-red-500 border border-red-500/20" :
                      result.risk_level === 'Medium' ? "bg-amber-500/10 text-amber-500 border border-amber-500/20" :
                      "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20"
                    )}>
                      {result.risk_level}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Likely Outcome</div>
                    <div className="text-slate-300 text-sm">{result.likely_outcome}</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="md:col-span-2 overflow-y-auto pr-2 custom-scrollbar">
            {!result && !isProcessing && (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-6">
                <div className="w-24 h-24 bg-white/5 rounded-3xl flex items-center justify-center border border-white/10 shadow-2xl">
                  <Search className="w-10 h-10 opacity-20" />
                </div>
                <div className="text-center">
                  <h3 className="text-white font-bold text-lg mb-2">Ready to Analyze</h3>
                  <p className="max-w-xs mx-auto text-sm">Input case details to find similar Supreme Court judgments and legal precedents.</p>
                </div>
              </div>
            )}

            {isProcessing && (
              <div className="h-full flex flex-col items-center justify-center gap-4">
                <Loader2 className="w-10 h-10 text-brand-blue animate-spin" />
                <p className="text-slate-400 font-medium animate-pulse">Searching 1,414 Supreme Court judgments...</p>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                <h3 className="text-white font-bold flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-brand-blue" />
                  Similar Precedents
                </h3>
                {result.similar_cases.map((c, idx) => (
                  <motion.div 
                    key={idx}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="glass-panel p-6 hover:border-white/10 transition-all"
                  >
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h4 className="text-lg font-bold text-white mb-1">{c.title}</h4>
                        <div className="flex items-center gap-2">
                          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                            {c.citation} • {c.court} • {c.year}
                          </div>
                          <button 
                            onClick={() => copyCitation(c.citation)}
                            className="p-1 hover:bg-white/5 rounded transition-colors text-slate-500 hover:text-brand-blue"
                            title="Copy Citation"
                          >
                            <Copy className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-black text-brand-blue">approximately {c.similarity}%</div>
                        <div className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">Similarity</div>
                      </div>
                    </div>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="bg-white/5 rounded-xl p-4 border border-white/5">
                        <div className="text-[9px] text-slate-500 font-bold uppercase tracking-widest mb-2">Relevance Reason</div>
                        <p className="text-slate-300 text-xs leading-relaxed mb-3">{c.relevance_reason}</p>
                        <div className="flex flex-wrap gap-2">
                          {c.legal_factors_matched.map((factor, i) => (
                            <span key={i} className="px-2 py-0.5 bg-brand-blue/10 text-brand-blue rounded text-[8px] font-bold uppercase tracking-widest border border-brand-blue/20">
                              {factor}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="bg-white/5 rounded-xl p-4 border border-white/5">
                        <div className="text-[9px] text-slate-500 font-bold uppercase tracking-widest mb-2">Verdict Summary</div>
                        <p className="text-slate-300 text-xs leading-relaxed">{c.verdict_summary}</p>
                        <div className="mt-3 inline-block px-2 py-1 bg-brand-blue/10 text-brand-blue rounded text-[9px] font-bold uppercase tracking-widest">
                          Result: {c.outcome}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const CitationVerification = () => {
  const [citation, setCitation] = useState('');
  const [result, setResult] = useState<CitationVerificationResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleVerify = async () => {
    if (!citation.trim()) return;
    setIsProcessing(true);
    try {
      const res = await verifyCitation(citation);
      setResult(res);
    } catch (error) {
      console.error("Verification Error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Valid': return 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20';
      case 'Overruled': return 'text-red-500 bg-red-500/10 border-red-500/20';
      case 'Distinguished': return 'text-amber-500 bg-amber-500/10 border-amber-500/20';
      case 'Caution': return 'text-orange-500 bg-orange-500/10 border-orange-500/20';
      default: return 'text-slate-500 bg-slate-500/10 border-slate-500/20';
    }
  };

  return (
    <div className="h-full flex flex-col bg-brand-dark p-8 overflow-hidden">
      <div className="max-w-6xl mx-auto w-full flex flex-col h-full">
        <div className="mb-10 flex justify-between items-end">
          <div>
            <h2 className="text-3xl font-bold text-white tracking-tight mb-2">Citation Verification</h2>
            <p className="text-slate-500 text-sm">Verify the current status and subsequent history of any Pakistani legal citation.</p>
          </div>
          {result && (
            <button 
              onClick={() => {setResult(null); setCitation('');}}
              className="btn-secondary text-xs px-6"
            >
              New Verification
            </button>
          )}
        </div>

        <div className="grid md:grid-cols-3 gap-8 flex-1 overflow-hidden">
          <div className="md:col-span-1 flex flex-col gap-6">
            <div className="glass-panel p-6">
              <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                <Search className="w-4 h-4 text-brand-blue" />
                Enter Citation
              </h3>
              <div className="space-y-4">
                <input 
                  type="text"
                  value={citation}
                  onChange={(e) => setCitation(e.target.value)}
                  placeholder="e.g. 2021 SCMR 123"
                  className="w-full bg-black/20 border border-white/5 rounded-xl p-4 text-slate-300 text-sm focus:outline-none focus:border-brand-blue/50 transition-all"
                />
                <button 
                  onClick={handleVerify}
                  disabled={isProcessing || !citation.trim()}
                  className="btn-primary w-full py-4 text-xs"
                >
                  {isProcessing ? (
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Verifying...
                    </div>
                  ) : "Verify Citation"}
                </button>
              </div>
            </div>

            {result && (
              <div className="glass-panel p-6">
                <h3 className="text-white font-bold mb-4 flex items-center gap-2">
                  <Info className="w-4 h-4 text-brand-blue" />
                  Case Info
                </h3>
                <div className="space-y-4">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Title</div>
                    <div className="text-white font-medium">{result.title}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Court</div>
                    <div className="text-white font-medium">{result.court}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Year</div>
                    <div className="text-white font-medium">{result.year}</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="md:col-span-2 overflow-y-auto pr-4 custom-scrollbar">
            {!result && !isProcessing && (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-6">
                <div className="w-24 h-24 bg-white/5 rounded-3xl flex items-center justify-center border border-white/10 shadow-2xl">
                  <ShieldCheck className="w-10 h-10 opacity-20" />
                </div>
                <div className="text-center">
                  <h3 className="text-white font-bold text-lg mb-2">Ready to Verify</h3>
                  <p className="max-w-xs mx-auto text-sm">Enter a citation to check if it's still valid or has been overruled by more recent judgments.</p>
                </div>
              </div>
            )}

            {isProcessing && (
              <div className="h-full flex flex-col items-center justify-center gap-4">
                <Loader2 className="w-10 h-10 text-brand-blue animate-spin" />
                <p className="text-slate-400 font-medium animate-pulse">Scanning Pakistani case law database...</p>
              </div>
            )}

            {result && (
              <div className="space-y-8">
                <div className="glass-panel p-8">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h3 className="text-2xl font-bold text-white mb-2">{result.citation}</h3>
                      {result.link && (
                        <a 
                          href={result.link} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-brand-blue text-xs font-bold flex items-center gap-1 hover:underline"
                        >
                          <ExternalLink className="w-3 h-3" />
                          View Official Judgment
                        </a>
                      )}
                    </div>
                    <div className={cn(
                      "px-4 py-1 rounded-full text-xs font-bold uppercase tracking-widest border",
                      getStatusColor(result.status)
                    )}>
                      {result.status}
                    </div>
                  </div>
                  
                  <div className="mb-8">
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-3">Judgment Summary</div>
                    <p className="text-slate-300 text-sm leading-relaxed">{result.summary}</p>
                  </div>

                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-3">Legal Analysis</div>
                    <div className="bg-white/5 rounded-xl p-4 border border-white/5">
                      <p className="text-slate-300 text-sm leading-relaxed">{result.analysis}</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="text-white font-bold flex items-center gap-2">
                    <History className="w-5 h-5 text-brand-blue" />
                    Subsequent History
                  </h3>
                  {result.subsequent_history.length > 0 ? (
                    result.subsequent_history.map((h, i) => (
                      <div key={i} className="glass-panel p-6">
                        <div className="flex justify-between items-center mb-3">
                          <div className="text-white font-bold">{h.citation}</div>
                          <div className={cn(
                            "px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest border",
                            h.treatment === 'Overruled' ? 'text-red-500 border-red-500/20 bg-red-500/5' : 'text-brand-blue border-brand-blue/20 bg-brand-blue/5'
                          )}>
                            {h.treatment}
                          </div>
                        </div>
                        <p className="text-slate-400 text-xs leading-relaxed">{h.reason}</p>
                      </div>
                    ))
                  ) : (
                    <div className="glass-panel p-8 text-center text-slate-500 text-sm italic">
                      No significant subsequent history found for this citation.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const UserDashboard = ({ onLogout }: { onLogout: () => void }) => {
  return (
    <div className="flex h-screen bg-brand-dark overflow-hidden font-sans">
      <main className="flex-1 flex flex-col">
        <header className="h-20 bg-brand-dark border-b border-white/5 flex items-center justify-between px-10 shrink-0">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-brand-blue/10 rounded-xl flex items-center justify-center border border-brand-blue/20">
              <Scale className="w-5 h-5 text-brand-blue" />
            </div>
            <div>
              <h2 className="font-bold text-white tracking-tight">Verdict AI</h2>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Public Desk</p>
            </div>
          </div>
          <button 
            onClick={onLogout}
            className="btn-secondary py-2 px-4 text-xs"
          >
            <LogOut className="w-4 h-4" />
            <span>Exit System</span>
          </button>
        </header>

        <div className="flex-1 overflow-hidden">
          <ChatInterface role="user" />
        </div>
      </main>
    </div>
  );
};

// --- Main App ---

export default function App() {
  const [role, setRole] = useState<UserRole | null>(null);

  const handleLogout = () => {
    setRole(null);
  };

  if (!role) {
    return <LandingPage onSelectRole={setRole} />;
  }

  return (
    <AnimatePresence mode="wait">
      {role === 'lawyer' ? (
        <motion.div 
          key="lawyer"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="h-screen"
        >
          <LawyerDashboard onLogout={handleLogout} />
        </motion.div>
      ) : role === 'deskaid' ? (
        <motion.div 
          key="deskaid"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="h-screen"
        >
          <DeskAidDashboard onLogout={handleLogout} />
        </motion.div>
      ) : (
        <motion.div 
          key="user"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="h-screen"
        >
          <UserDashboard onLogout={handleLogout} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
