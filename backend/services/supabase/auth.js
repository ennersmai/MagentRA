import { createClient } from '@supabase/supabase-js'

// Verify environment variables
console.log('Supabase URL:', process.env.SUPABASE_URL)
console.log('Supabase Key:', process.env.SUPABASE_ANON_KEY?.slice(0, 6) + '...')

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY,
  {
    auth: {
      persistSession: false,
      autoRefreshToken: false
    },
    db: {
      schema: 'public'
    }
  }
)

// Test connection
supabase.from('documents').select('*', { count: 'exact' })
  .then(({ error }) => {
    if (error) {
      console.error('Supabase connection test failed:', error)
      process.exit(1)
    }
    console.log('Supabase connection verified')
  })

export default supabase;
  