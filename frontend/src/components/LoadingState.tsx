import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  progress: number;
  message: string;
}

export default function LoadingState({ progress, message }: Props) {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col items-center justify-center text-center w-full max-w-lg"
    >
      <div className="w-full h-1 bg-gray-200 rounded-full overflow-hidden mb-10 relative">
        <motion.div 
          className="absolute left-0 top-0 bottom-0 bg-finance-ink"
          initial={{ width: "0%" }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>

      <h2 className="font-serif text-3xl text-finance-ink mb-6 font-bold tracking-wide">
        {Math.round(progress)}%
      </h2>
      
      <div className="h-8 overflow-hidden relative w-full">
        <AnimatePresence mode='wait'>
          <motion.p
            key={message}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -20, opacity: 0 }}
            className="text-finance-subtle font-mono text-sm w-full absolute top-0 left-0 right-0"
          >
            {message}
          </motion.p>
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
