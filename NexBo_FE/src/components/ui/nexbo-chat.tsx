"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { Textarea } from "./textarea";
import { cn } from "@/lib/utils";
import {
    TrendingUp,
    CircleUserRound,
    ArrowUpIcon,
    ChevronDown,
    ChevronUp,
    DollarSign,
    LineChart,
    PieChart,
    Wallet,
    ArrowLeft,
} from "lucide-react";
import { BackgroundBeams } from "@/components/ui/background-beams";
import { Button } from "@/components/ui/button";
import { NexBoLogo } from "./nexbo-logo";
import { useRouter } from "next/navigation";
import { toast, Toaster } from "sonner";
import ReactMarkdown from "react-markdown";

interface Stock {
    symbol: string;
    name: string;
    priority: number;
    score: string;
    price: string;
    shares: number;
    recommendation: string;
    analysis: string;
}

// Dummy suggestion queries
const suggestionQueries = [
    { icon: <DollarSign className="w-4 h-4" />, label: "Stock investment tips for beginners" },
    { icon: <LineChart className="w-4 h-4" />, label: "Myth Busting in the Stock Market" },
    { icon: <PieChart className="w-4 h-4" />, label: "Correct attitude towards investing" },
    { icon: <Wallet className="w-4 h-4" />, label: "Long-term investment advice" },
];

// Initial welcome message
const initialMessages: Message[] = [
    {
        role: 'assistant',
        content: "Welcome to NexBo!\nI'm your personal AI financial advisor, ready to assist you with:\n• Personalized stock recommendations\n• Investment strategies\n• Market analysis\n• Portfolio management\nTo get started, you can:\n1. Ask me any finance-related question\n2. Toggle 'Recommendations Mode' for stock picks\n3. Try the suggestion queries below"
    }
];

interface Message {
    role: 'user' | 'assistant';
    content: string;
    recommendations?: Stock[];
    timestamp?: Date;
}

interface StockCardProps {
    stock: Stock;
    expanded: boolean;
    onToggle: () => void;
}

const StockCard: React.FC<StockCardProps> = ({ stock, expanded, onToggle }) => (
    <div className="bg-white/5 p-4 rounded-xl space-y-2 transition-all duration-300 hover:bg-white/10 border border-white/10">
        <div className="flex items-center justify-between">
            <h4 className="font-semibold text-white flex items-center gap-2">
                {stock.symbol}
                <span className="text-xs font-normal text-white/60">{stock.name}</span>
            </h4>
            <span className={cn(
                "text-sm font-medium px-2 py-1 rounded-full",
                stock.score.startsWith("+")
                    ? "bg-green-500/20 text-green-400"
                    : "bg-red-500/20 text-red-400"
            )}>
                {stock.score}
            </span>
        </div>

        <div className="flex items-center justify-between">
            <p className="text-lg font-medium text-white">{stock.price}</p>
            <p className="text-xs text-white/60">{stock.shares} shares</p>
            <div className="text-xs px-2 py-1 rounded-full bg-blue-500/20 text-blue-400">
                Rank #{stock.priority}
            </div>
        </div>

        <div className={cn(
            "grid transition-all duration-300",
            expanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
        )}>
            <div className="overflow-hidden">
                <div className="pt-2 space-y-2 border-t border-white/10">
                    <div className="text-sm font-medium text-white/90 flex items-center gap-2">
                        <span className={cn(
                            "px-2 py-1 rounded-full text-xs",
                            stock.recommendation === "Strong Buy"
                                ? "bg-green-500/20 text-green-400"
                                : stock.recommendation === "Buy"
                                    ? "bg-emerald-500/20 text-emerald-400"
                                    : "bg-yellow-500/20 text-yellow-400"
                        )}>
                            {stock.recommendation}
                        </span>
                    </div>
                    <p className="text-sm text-white/70 leading-relaxed">{stock.analysis}</p>
                </div>
            </div>
        </div>

        <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className="w-full mt-2 text-white/60 hover:text-white group"
        >
            {expanded ? (
                <span className="flex items-center gap-2">
                    <ChevronUp className="h-4 w-4" />
                    <span className="text-xs group-hover:text-white/90">Show Less</span>
                </span>
            ) : (
                <span className="flex items-center gap-2">
                    <ChevronDown className="h-4 w-4" />
                    <span className="text-xs group-hover:text-white/90">View Analysis</span>
                </span>
            )}
        </Button>
    </div>
);

function useAutoResizeTextarea({
    minHeight,
    maxHeight,
}: {
    minHeight: number;
    maxHeight?: number;
}) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const adjustHeight = useCallback(
        (reset?: boolean) => {
            const textarea = textareaRef.current;
            if (!textarea) return;

            if (reset) {
                textarea.style.height = `${minHeight}px`;
                return;
            }

            textarea.style.height = `${minHeight}px`;
            const newHeight = Math.max(
                minHeight,
                Math.min(
                    textarea.scrollHeight,
                    maxHeight ?? Number.POSITIVE_INFINITY
                )
            );
            textarea.style.height = `${newHeight}px`;
        },
        [minHeight, maxHeight]
    );

    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = `${minHeight}px`;
        }
    }, [minHeight]);

    useEffect(() => {
        const handleResize = () => adjustHeight();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, [adjustHeight]);

    return { textareaRef, adjustHeight };
}

export function NexBoChat() {
    const router = useRouter();
    const [value, setValue] = useState("");
    const [messages, setMessages] = useState<Message[]>(initialMessages);
    const [isRecommendationMode, setIsRecommendationMode] = useState(false);
    const [expandedStocks, setExpandedStocks] = useState<Record<string, boolean>>({});
    const [isWaitingForParameters, setIsWaitingForParameters] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const chatContainerRef = useRef<HTMLDivElement>(null);

    const { textareaRef, adjustHeight } = useAutoResizeTextarea({
        minHeight: 60,
        maxHeight: 200,
    });

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const toggleStockExpansion = (symbol: string) => {
        setExpandedStocks(prev => ({
            ...prev,
            [symbol]: !prev[symbol]
        }));
    };

    const handleToggleRecommendation = () => {
        const newMode = !isRecommendationMode;
        setIsRecommendationMode(newMode);

        if (newMode) {
            // Recommendation mode turned ON
            toast.success("Recommendation Engine Activated", {
                description: "NexBo will now provide personalized stock recommendations",
                duration: 3000,
            });

            // Ask for parameters
            setIsWaitingForParameters(true);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: "To provide personalized stock recommendations, I need some information. Please provide:\n\n" +
                    "1. Budget: (e.g., $10,000)\n" +
                    "2. Preferred sectors (Technology/Healthcare/Finance/Energy/Consumer)\n" +
                    "3. Market (US/India)\n\n"
            }]);
        } else {
            // Recommendation mode turned OFF
            toast.info("Recommendation Engine Deactivated", {
                description: "NexBo will no longer provide stock recommendations",
                duration: 3000,
            });
            setIsWaitingForParameters(false);
        }
    };

    const handleSend = async () => {
        if (!value.trim() || isLoading) return;

        // Store the message text before clearing input
        const messageText = value.trim();
        
        // Clear input immediately for better UX
        setValue("");
        
        // Add user message to chat
        const userMessage: Message = {
            role: 'user',
            content: messageText,
            timestamp: new Date()
        };
        setMessages(prev => [...prev, userMessage]);

        try {
            setIsLoading(true);
            
            console.log("💬 Sending message:", {
                message: messageText,
                recommend: isRecommendationMode
            });

            const response = await fetch('https://dark-matter-nexbo.onrender.com/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: messageText,
                    recommend: isRecommendationMode
                }),
            });

            const data = await response.json();
            console.log("📡 Received response:", data);

            if (data.error) {
                console.error("❌ Error:", data.error);
                toast.error(data.error);
                return;
            }

            // Add assistant response with recommendations if present
            const aiMessage: Message = {
                role: 'assistant',
                content: data.response || '',
                recommendations: Array.isArray(data.recommendations) ? data.recommendations : []
            };

            console.log("🤖 AI Message with recommendations:", aiMessage);
            setMessages(prev => [...prev, aiMessage]);

        } catch (error) {
            console.error('❌ Chat error:', error);
            toast.error('Failed to get response');
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <>
            <div className="flex flex-col h-screen bg-black">
                <BackgroundBeams className="opacity-20" />
                <Toaster position="top-center" />

                {/* Header */}
                <div className="relative border-b border-white/10 bg-black/40 backdrop-blur-sm">
                    <div className="max-w-5xl mx-auto px-4 py-3 flex items-center">
                        {/* Back Button */}
                        <button
                            aria-label="Go back"
                            onClick={() => router.back()}
                            className="p-2 rounded-lg bg-white/5 border border-white/10 text-white/60 hover:bg-white/10 hover:text-white transition-colors"
                        >
                            <ArrowLeft className="w-5 h-5" />
                        </button>

                        {/* Centered Title */}
                        <div className="flex-1 flex justify-center items-center">
                            <div className="flex items-center gap-3">
                                <NexBoLogo size="sm" />
                                <div>
                                    <h1 className="text-lg font-semibold text-white">NexBo Chat</h1>
                                    <p className="text-xs text-white/60">AI-Powered Financial Assistant</p>
                                </div>
                            </div>
                        </div>

                        {/* Empty div for spacing */}
                        <div className="w-8"></div>
                    </div>
                </div>

                {/* Chat container */}
                <div className="relative flex-1 overflow-hidden" ref={chatContainerRef}>
                    <div className="absolute inset-0 overflow-y-auto p-4 space-y-6 hide-scrollbar">
                        {messages.map((message, index) => (
                            <div key={index}>
                                <div
                                    className={cn(
                                        "max-w-4xl mx-auto p-4 rounded-xl backdrop-blur-sm flex",
                                        message.role === 'user'
                                            ? "justify-end" // Removed border from user messages
                                            : "bg-zinc-900/50 border border-white/10"
                                    )}
                                >
                                    <div className={cn(
                                        "flex items-start gap-3",
                                        message.role === 'user' ? "flex-row-reverse" : ""
                                    )}>
                                        {message.role === 'assistant' ? (
                                            <NexBoLogo
                                         size="sm" />
                                        ) : (
                                            <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
                                                <CircleUserRound className="w-5 h-5 text-blue-400" />
                                            </div>
                                        )}
                                        <div className={cn(
                                            "space-y-4",
                                            message.role === 'user' ? "text-right" : ""
                                        )}>
                                            {message.role === 'assistant'?(
                                                    <ReactMarkdown
                                                        className="text-white/90 leading-relaxed rounded-xl p-3 whitespace-pre-line ai-response-bg"
                                                    >
                                                        {message.content}
                                                    </ReactMarkdown>

                                            ) :(
                                            <p className={cn(
                                                "text-white/90 leading-relaxed whitespace-pre-line rounded-xl p-3",
                                                message.role === 'user' ? "bg-blue-500/20" : ""
                                            )}>
                                                {message.content}
                                            </p>
                                            )}
                                            {message.recommendations && (
                                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                                                    {message.recommendations.map((stock) => (
                                                        <StockCard
                                                            key={stock.symbol}
                                                            stock={stock}
                                                            expanded={!!expandedStocks[stock.symbol]}
                                                            onToggle={() => toggleStockExpansion(stock.symbol)}
                                                        />
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                                {isLoading && index === messages.length - 1 && message.role === 'user' && (
                                    <div className="max-w-4xl mx-auto p-4 flex">
                                        <div className="flex items-center gap-2">
                                            <NexBoLogo size="sm" />
                                            <div className="flex items-center gap-2">
                                                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                                                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                                                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}                        <div ref={messagesEndRef} />
                    </div>
                </div>

                {/* Input area */}
                <div className="relative p-4 bg-black/50 border-t border-white/10 backdrop-blur-sm">
                    <div className="max-w-4xl mx-auto">
                        {/* Suggestion queries - MOVED HERE */}
                        <div className="flex items-center gap-2 mb-3 overflow-x-auto pb-2 hide-scrollbar">
                            {suggestionQueries.map((query, index) => (
                                <button
                                    key={index}
                                    onClick={() => setValue(query.label)}
                                    className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-full border border-white/10 text-white/80 hover:text-white transition-colors whitespace-nowrap"
                                >
                                    {query.icon}
                                    <span className="text-xs">{query.label}</span>
                                </button>
                            ))}
                        </div>

                        {/* Input Box and Controls */}
                        <div className="relative bg-zinc-900/80 rounded-xl border border-white/10">
                            <Textarea
                                ref={textareaRef}
                                value={value}
                                onChange={(e) => {
                                    setValue(e.target.value);
                                    adjustHeight();
                                }}
                                onKeyDown={handleKeyDown}
                                placeholder={isWaitingForParameters
                                    ? "Enter your investment preferences..."
                                    : "Ask NexBo about finance..."}
                                className={cn(
                                    "w-full px-4",
                                    "resize-none",
                                    "bg-transparent",
                                    "border-none",
                                    "text-white text-sm",
                                    "focus:outline-none",
                                    "focus-visible:ring-0 focus-visible:ring-offset-0",
                                    "placeholder:text-zinc-500 placeholder:text-sm",
                                    "min-h-[40px]",
                                    "flex items-center",
                                    "py-[21px]"
                                )}
                                style={{
                                    overflow: "hidden",
                                }}
                            />

                            <div className="flex items-center justify-end gap-2 p-3 border-t border-white/10">
                                <Button
                                    onClick={handleToggleRecommendation}
                                    className={cn(
                                        "transition-all duration-300",
                                        isRecommendationMode
                                            ? "bg-gradient-to-br from-zinc-400 to-zinc-600 hover:from-zinc-500 hover:to-zinc-700 text-white"
                                            : "bg-white/5 hover:bg-white/10 text-white/80 hover:text-white border border-white/20"
                                    )}
                                >
                                    <TrendingUp className={cn(
                                        "w-4 h-4 mr-2",
                                        isRecommendationMode ? "text-white" : "text-white/80"
                                    )} />
                                    {isRecommendationMode ? "Recommendations: ON" : "Recommendations: OFF"}
                                </Button>

                                <button
                                    onClick={handleSend}
                                    type="button"
                                    disabled={!value.trim() || isLoading}
                                    className={cn(
                                        "p-3 rounded-full transition-all duration-300",
                                        value.trim() && !isLoading
                                            ? "bg-blue-500 hover:bg-blue-600 text-white"
                                            : "text-white/40 bg-white/5 cursor-not-allowed"
                                    )}
                                >
                                    {isLoading ? (
                                        <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                                    ) : (
                                        <ArrowUpIcon
                                            className={cn(
                                                "w-5 h-5 transition-transform duration-300",
                                                value.trim() && "transform -translate-y-0.5"
                                            )}
                                        />
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
