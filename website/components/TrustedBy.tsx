"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import Container from "./Container";

export default function TrustedBy() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });

  const companies = [
    "Dell Technologies",
    "Oracle",
    "SAP",
    "Amazon",
    "Ellipsis Health",
  ];

  return (
    // Commented out entire Trusted By section
    null
    /* <section className="py-16 bg-gradient-to-b from-white to-gray-50 dark:from-black dark:to-gray-900/50 border-y border-gray-200 dark:border-gray-800" ref={ref}>
      <Container>
        {/* Commented out Trusted By section
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <p className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-8">
            Trusted by developers at
          </p>

          <div className="flex flex-wrap items-center justify-center gap-8 md:gap-12">
            {companies.map((company, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={inView ? { opacity: 1, scale: 1 } : {}}
                transition={{ duration: 0.5, delay: index * 0.1, type: "spring", stiffness: 300 }}
                whileHover={{ scale: 1.15, y: -8 }}
                className="group relative"
              >
                <div className="absolute -inset-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl opacity-0 group-hover:opacity-30 blur-2xl transition-opacity duration-500" />

                <motion.div
                  className="absolute inset-0 rounded-2xl bg-gradient-to-r from-green-500 to-emerald-500 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                  style={{ padding: "2px" }}
                >
                  <div className="absolute inset-[2px] rounded-2xl bg-white dark:bg-gray-800" />
                </motion.div>

                <div className="relative px-8 py-4 rounded-2xl bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm shadow-lg group-hover:shadow-2xl transition-all border-2 border-transparent">
                  <motion.span
                    className="text-lg font-bold text-gray-800 dark:text-gray-200 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-green-500 group-hover:to-emerald-500 transition-all relative z-10"
                    whileHover={{ scale: 1.05 }}
                  >
                    {company}
                  </motion.span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
        *
      </Container>
    </section> */
  );
}
